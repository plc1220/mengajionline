import asyncio
import base64
import io
import os
import pandas as pd
import streamlit as st
import time
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from datetime import datetime, timezone
from google.api_core.client_options import ClientOptions
from google.cloud import bigquery
from google.cloud import discoveryengine_v1beta as discoveryengine
from google.cloud import storage
from langchain_text_splitters import RecursiveCharacterTextSplitter

from pandas_gbq import to_gbq
from pydub import AudioSegment
from pydub.silence import detect_nonsilent
from streamlit.logger import get_logger
from tenacity import retry, stop_after_attempt, wait_exponential
from vertexai.generative_models import GenerativeModel, Part
from proto import Message

PAGE_TITLE = "Video Transcription and Search"
PAGE_INFO = """
This tool enables efficient transcription, summarization, and search of video content. \
By extracting audio from videos and segmenting it into manageable chunks, \
the tool uses Gemini’s advanced multimodal understanding to transcribe the audio accurately. \
The transcription is automatically summarized and indexed, making it easy to search and retrieve relevant video segments.

### How to Use This Tool:

1. **Search**:  
   - Enter a search query and specify the number of results to return.
   - The tool retrieves relevant segments based on your query.
   - You can download the transcript for each segment.

2. **Transcribe**:  
   - Choose from sample videos or upload your own.
   - The tool begins transcribing; the time taken depends on the video's length.
   - Optionally edit the transcripts before saving them to the search index for future retrieval.
"""


PROJECT_ID = os.environ.get("PROJECT_ID")
LOCATION = os.environ.get("LOCATION", "global")
DATA_STORE_ID = os.environ.get("DATA_STORE_ID")
RAW_TRANSCRIPT_BQ_TABLE_ID = os.environ.get(
    "RAW_TRANSCRIPT_BQ_TABLE_ID"
)
STAGING_TRANSCRIPT_BQ_TABLE_ID = os.environ.get(
    "STAGING_TRANSCRIPT_BQ_TABLE_ID",
)
SUMMARY_BQ_TABLE_ID = os.environ.get("SUMMARY_BQ_TABLE_ID")
SOURCE_FOLDER = os.environ.get("SOURCE_FOLDER", "app/data/video/")

logger = get_logger(__name__)


def set_model():
    model_name = st.session_state.get("model_name", "gemini-2.0-flash-001")
    model = GenerativeModel(
        model_name=model_name,
        generation_config={"max_output_tokens": 8192, "temperature": 0},
    )
    st.session_state["model"] = model


def list_video_files(folder_path: str) -> list[str]:
    video_extensions = (".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv")
    video_files = []

    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(video_extensions):
                video_files.append(os.path.join(root, file))

    return video_files


def upload_video(file):
    # Ensure directory exists
    os.makedirs(SOURCE_FOLDER, exist_ok=True)
    video_file_path = os.path.join(SOURCE_FOLDER, file.name)
    with open(video_file_path, "wb") as f:
        f.write(file.getvalue())
    return video_file_path


def load_and_split_audio(video_path, min_silence_len=1000, silence_thresh=-40, min_segment_len=30000) -> list[str]:
    # Extract audio from video
    audio = AudioSegment.from_file(video_path)

    # Detect non-silent segments
    nonsilent_ranges = detect_nonsilent(
        audio, min_silence_len=min_silence_len, silence_thresh=silence_thresh, seek_step=50
    )

    # Combine short segments and split audio
    combined_segments = []
    current_segment = nonsilent_ranges[0]
    for next_segment in nonsilent_ranges[1:]:
        if current_segment[1] - current_segment[0] < min_segment_len:
            # Combine segments
            current_segment = (current_segment[0], next_segment[1])
        else:
            # Add current segment and start a new one
            combined_segments.append(current_segment)
            current_segment = next_segment

    # Add the last segment
    combined_segments.append(current_segment)

    # Export combined segments
    audio_segments = []
    for i, (start_ms, end_ms) in enumerate(combined_segments):
        # Extract segment with buffer
        start = int(max(0, start_ms - 200))
        end = int(min(audio.duration_seconds * 1000, end_ms + 1000))
        segment = audio[start:end]

        # Export segment with timestamp in filename
        buffer = io.BytesIO()
        segment.export(buffer, format="mp3")
        audio_data = buffer.getvalue()
        base64_audio = base64.b64encode(audio_data).decode("utf-8")

        audio_segments.append({"index": i, "start_ms": start, "end_ms": end, "base64_audio": base64_audio})

    return audio_segments


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
async def generate_content_with_retry(base64_audio: str):
    prompt = """
    Transcribe this audio clip. Use proper casing and punctuations. There could be more than one speaker in the clip. Distinguish each speaker by their voice, and assign caption to the speakers, using this format:
  
    - Speaker 1: caption.

    - Speaker 2: caption.

    ...

    - Speaker N: caption.

    Use Speaker 1, Speaker 2, Speaker 3, etc. to identify the different speakers. Respond in markdown format.
    """
    audio_file = Part.from_data(data=base64_audio, mime_type="audio/mpeg")
    contents = [audio_file, prompt]
    return await st.session_state["model"].generate_content_async(contents)


async def generate_transcript(audio_segment: dict, semaphore) -> str:
    async with semaphore:
        try:
            response = await generate_content_with_retry(audio_segment["base64_audio"])
            logger.info(response.usage_metadata)
            audio_segment["transcript"] = response.text
        except Exception as e:
            logger.error(f"Error processing segment {audio_segment['index']}: {str(e)}")
            audio_segment["transcript"] = None

        return audio_segment


async def generate_transcript_batch(audio_segments: list[dict], concurrency=8):
    semaphore = asyncio.Semaphore(concurrency)
    tasks = [generate_transcript(segment, semaphore) for segment in audio_segments]
    return await asyncio.gather(*tasks)


def format_timestamp(ms):
    """
    Convert milliseconds to HH:MM:SS format.
    """
    seconds = int(ms / 1000)
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def split_text(text: str) -> list[str]:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=100, separators=[". "], is_separator_regex=False, keep_separator=False
    )
    return text_splitter.split_text(text)


def refresh_data_store():
    try:
        # Log configuration for debugging
        logger.info(f"Refresh Data Store Configuration:")
        logger.info(f"PROJECT_ID: {PROJECT_ID}")
        logger.info(f"LOCATION: {LOCATION}")
        logger.info(f"DATA_STORE_ID: {DATA_STORE_ID}")
        logger.info(f"STAGING_TRANSCRIPT_BQ_TABLE_ID: {STAGING_TRANSCRIPT_BQ_TABLE_ID}")
        
        # Discovery Engine API endpoint format changed to include the protocol
        api_endpoint = f"https://{LOCATION}-discoveryengine.googleapis.com"
        client_options = ClientOptions(api_endpoint=api_endpoint) if LOCATION != "global" else None
        logger.info(f"API Endpoint: {api_endpoint}")
        
        client = discoveryengine.DocumentServiceClient(client_options=client_options)

        parent = client.branch_path(
            project=PROJECT_ID,
            location=LOCATION,
            data_store=DATA_STORE_ID,
            branch="default_branch",
        )
        logger.info(f"Parent path: {parent}")

        request = discoveryengine.ImportDocumentsRequest(
            parent=parent,
            bigquery_source=discoveryengine.BigQuerySource(
                dataset_id=STAGING_TRANSCRIPT_BQ_TABLE_ID.split(".")[-2],
                table_id=STAGING_TRANSCRIPT_BQ_TABLE_ID.split(".")[-1],
                data_schema="custom",
            ),
            reconciliation_mode=discoveryengine.ImportDocumentsRequest.ReconciliationMode.FULL,
            auto_generate_ids=True,
        )

        operation = client.import_documents(request=request)
        response = operation.result()
        metadata = discoveryengine.ImportDocumentsMetadata(operation.metadata)
        logger.info(f"Operation name: {operation.operation.name}")
        logger.info(f"Metadata: {metadata}")
        return metadata.success_count
    except Exception as e:
        logger.error(f"Error refreshing data store: {str(e)}")
        st.error(f"Error refreshing search index: {str(e)}")
        return 0


def edit_segment(index):
    st.session_state["transcribed_segments"][index]["edit"] = True


def update_segment(index):
    st.session_state["transcribed_segments"][index]["transcript"] = st.session_state[f"transcript_{index}"]
    st.session_state["transcribed_segments"][index]["edit"] = False


def save_segment(index):
    st.session_state["transcribed_segments"][index]["done"] = True


def get_transcripts():
    bq_client = bigquery.Client()
    # Get transcripts
    sql = f"SELECT index as segment_id, video_file_path, transcript FROM {STAGING_TRANSCRIPT_BQ_TABLE_ID}"
    df_transcript = bq_client.query(sql).to_dataframe()

    # Get summary
    sql = f"""SELECT video_file_path, summary
        FROM (
            SELECT
                *,
                ROW_NUMBER() OVER (PARTITION BY video_file_path ORDER BY created_at DESC) AS row_num
            FROM `{SUMMARY_BQ_TABLE_ID}`
        )
        WHERE row_num = 1
        """
    df_summary = bq_client.query(sql).to_dataframe()

    st.session_state["n_videos"] = df_transcript["video_file_path"].nunique()
    st.session_state["full_transcripts"] = (
        df_transcript.sort_values(by="segment_id").groupby("video_file_path")["transcript"].apply("\n".join).to_dict()
    )
    st.session_state["summaries"] = df_summary.groupby("video_file_path")["summary"].first().to_dict()
    logger.info(st.session_state["summaries"])


def search():
    search_query = st.session_state.get("search_query", None)
    max_results = st.session_state.get("max_results", 10)
    if search_query:
        try:
            # Log configuration for debugging
            logger.info(f"Search Configuration:")
            logger.info(f"PROJECT_ID: {PROJECT_ID}")
            logger.info(f"LOCATION: {LOCATION}")
            logger.info(f"DATA_STORE_ID: {DATA_STORE_ID}")
            
            # Discovery Engine API endpoint format changed to include the protocol
            api_endpoint = f"https://{LOCATION}-discoveryengine.googleapis.com"
            client_options = ClientOptions(api_endpoint=api_endpoint) if LOCATION != "global" else None
            logger.info(f"API Endpoint: {api_endpoint}")
            
            client = discoveryengine.SearchServiceClient(client_options=client_options)
            serving_config = f"projects/{PROJECT_ID}/locations/{LOCATION}/collections/default_collection/dataStores/{DATA_STORE_ID}/servingConfigs/default_config"
            logger.info(f"Serving Config: {serving_config}")

            request = discoveryengine.SearchRequest(
                serving_config=serving_config, query=search_query, page_size=max_results
            )

            response = client.search(request)

            results = []
            for result in response:
                results.append(Message.to_dict(result)["document"]["struct_data"])
                if len(results) == max_results:
                    break
            st.session_state["search_results"] = results
            
        except Exception as e:
            logger.error(f"Search error: {str(e)}")
            st.error(f"Error performing search: {str(e)}")


def summarize(text):
    response = st.session_state["model"].generate_content(f"Summarize this video transcript: {text}")
    return response.text


def display_search_result(index, result):
    label = f"**Source:** {result['video_file_path'].split('/')[-1]}  ({format_timestamp(result['start_ms'])} - {format_timestamp(result['end_ms'])})"
    with st.expander(label, expanded=False):
        st.markdown(result["transcript"])
        cols = st.columns(3)
        if cols[0].button(
            ":material/play_arrow: Play video",
            key=f"play_video_{index}",
            use_container_width=True,
        ):
            st.video(result["video_file_path"], start_time=result["start_ms"] // 1000)
        cols[1].download_button(
            ":material/download: Download summary",
            key=f"summary_{index}",
            data=st.session_state["summaries"][result["video_file_path"]],
            file_name=f"{result['video_file_path'].split('/')[-1].split('.')[0]}_summary.txt",
            use_container_width=True,
        )
        cols[2].download_button(
            ":material/download: Download full transcript",
            key=f"download_{index}",
            data=st.session_state["full_transcripts"][result["video_file_path"]],
            file_name=f"{result['video_file_path'].split('/')[-1].split('.')[0]}_transcript.txt",
            use_container_width=True,
        )


def update_staging_table():
    bq_client = bigquery.Client()
    sql = f"""
    CREATE OR REPLACE TABLE `{STAGING_TRANSCRIPT_BQ_TABLE_ID}` AS
    (
        SELECT t1.* FROM `{RAW_TRANSCRIPT_BQ_TABLE_ID}` t1
        INNER JOIN (
            SELECT
                video_file_path,
                MAX(created_at) AS max_created_at
            FROM `{RAW_TRANSCRIPT_BQ_TABLE_ID}`
            GROUP BY 1 ) t2
        ON
        t1.video_file_path = t2.video_file_path
        AND t1.created_at = t2.max_created_at
    )
    """
    bq_client.query_and_wait(query=sql)


def save_summary():
    bq_client = bigquery.Client()
    data = [
        {
            "video_file_path": st.session_state["video_file_path"],
            "summary": st.session_state["transcript_summary"],
            "created_at": str(datetime.now(timezone.utc)),
        }
    ]
    bq_client.insert_rows_json(SUMMARY_BQ_TABLE_ID, data)
    st.toast(f"Saved video summary", icon="✅")


def save_transcript():
    with st.spinner("Saving transcripts to the data store..."):
        df = pd.DataFrame(
            st.session_state["transcribed_segments"], columns=["index", "start_ms", "end_ms", "transcript"]
        )
        df["video_file_path"] = st.session_state["video_file_path"]
        df["created_at"] = datetime.now(timezone.utc)

        # Insert chunks into BQ
        try:
            to_gbq(df, RAW_TRANSCRIPT_BQ_TABLE_ID, if_exists="append")
            update_staging_table()
            st.toast("Saved transcripts to the data store", icon="✅")
        except Exception as e:
            st.toast("Encountered errors while saving transcripts: {}".format(str(e)), icon="❌")

    # Refresh index
    with st.spinner("Refreshing search index..."):
        refresh_data_store()
    st.toast(f"Refreshed index", icon="✅")

    # Clear session state
    # for key in st.session_state.keys():
    #     del st.session_state[key]


# Initialize model
if "model" not in st.session_state:
    model = GenerativeModel(
        model_name="gemini-2.0-flash-001",
        generation_config={"max_output_tokens": 8192, "temperature": 0},
    )
    st.session_state["model"] = model

tab1, tab2 = st.tabs(["Search", "Transcribe"])

with tab1:
    if "n_videos" not in st.session_state:
        get_transcripts()

    with st.form("submit_form", clear_on_submit=False):
        st.info(f"Total number of videos transcribed: **{st.session_state['n_videos']}**", icon=":material/info:")
        cols = st.columns(2)
        with cols[0]:
            query = st.text_input("Search query", key="search_query")
        with cols[1]:
            max_results = st.number_input("Number of results", key="max_results", min_value=1, max_value=20, value=5)

        st.form_submit_button(":material/search: Search", on_click=search, type="primary")

    if "search_results" in st.session_state:
        st.subheader("Search results")
        for i, result in enumerate(st.session_state["search_results"]):
            display_search_result(i, result)

with tab2:
    # User input
    with st.container(border=True):
        cols = st.columns(2)
        with cols[0]:
            video_files = list_video_files(SOURCE_FOLDER)
            video_path = st.selectbox(
                "Select from sample video files",
                video_files,
                index=None,
                key="selected_video_file",
                format_func=lambda x: x.split("/")[-1],
            )

        with cols[1]:
            model_name = st.selectbox(
                "Pick a model",
                ["gemini-2.0-flash-001", "gemini-2.0-pro-exp-02-05"],
                index=0,
                key="model_name",
                on_change=set_model,
            )

        st.file_uploader("[Optional] Upload a video", type=["mp4"], key="uploaded_video")

        if st.session_state.get("uploaded_video"):
            st.video(st.session_state["uploaded_video"])

        elif st.session_state.get("selected_video_file"):
            st.video(st.session_state["selected_video_file"])

        submit_video = st.button(":material/transcribe: Transcribe", type="primary")

    # Transcribe new video input
    if submit_video:
        if st.session_state.get("uploaded_video"):
            with st.spinner("Uploading video..."):
                video_file_path = upload_video(st.session_state["uploaded_video"])
                st.session_state["video_file_path"] = video_file_path
        elif st.session_state.get("selected_video_file"):
            st.session_state["video_file_path"] = st.session_state["selected_video_file"]
        else:
            st.error("Please select a video to transcribe", icon="❌")
            st.stop()

        with st.spinner("Extracting audio..."):
            audio_segments = load_and_split_audio(
                st.session_state["video_file_path"],
                min_silence_len=1200,
                silence_thresh=-45,
                min_segment_len=60000,
            )

        with st.spinner("Transcribing..."):
            transcribed_segments = asyncio.run(generate_transcript_batch(audio_segments, concurrency=8))
            st.session_state["transcribed_segments"] = sorted(transcribed_segments, key=lambda x: x["index"])

        with st.spinner("Summarizing..."):
            summary = summarize(
                "\n".join([segment["transcript"] for segment in st.session_state["transcribed_segments"]])
            )
            st.session_state["transcript_summary"] = summary

    if "transcript_summary" in st.session_state:
        st.subheader("Summary")
        st.markdown(st.session_state["transcript_summary"])
        st.button(":material/save: Save summary", use_container_width=True, on_click=save_summary)

    if "transcribed_segments" in st.session_state:
        st.subheader("Transcripts")
        for segment in st.session_state["transcribed_segments"]:
            label = f"Segment {segment['index']+1} [{format_timestamp(segment['start_ms'])} - {format_timestamp(segment['end_ms'])}]"
            if segment.get("done", False):
                label = "✔ " + label
            with st.expander(label, expanded=False):
                st.audio(base64.b64decode(segment["base64_audio"].encode("utf-8")), format="audio/mpeg")
                if segment.get("edit", False):
                    st.text_area(
                        "Edit transcript",
                        value=segment["transcript"],
                        key=f"transcript_{segment['index']}",
                        height=300,
                    )
                    cols = st.columns(4)
                    cols[3].button(
                        "Done",
                        key=f"save_button_{segment['index']}",
                        on_click=update_segment,
                        args=(segment["index"],),
                        use_container_width=True,
                    )
                else:
                    st.markdown(segment["transcript"])
                    cols = st.columns(4)
                    cols[2].button(
                        "Edit",
                        key=f"edit_button_{segment['index']}",
                        on_click=edit_segment,
                        args=(segment["index"],),
                        use_container_width=True,
                    )
                    cols[3].button(
                        "Mark as done",
                        key=f"save_button_{segment['index']}",
                        on_click=save_segment,
                        args=(segment["index"],),
                        use_container_width=True,
                    )
        st.button(":material/save: Save transcripts", use_container_width=True, on_click=save_transcript)
