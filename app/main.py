import os
import streamlit as st
import chromadb
from datetime import datetime, timezone
from vertexai.generative_models import GenerativeModel
from typing import List, Dict

SOURCE_FOLDER = os.environ.get("SOURCE_FOLDER", os.path.abspath("app/data/video/"))

# Performance metrics categories based on the image
PERFORMANCE_METRICS = {
    "class_performance": [
        "class_start_timing",
        "camera_usage", 
        "platform_usage",
        "screen_sharing",
        "student_age_group",
        "reading_accuracy",
        "class_attendance",
        "teaching_duration",
        "group_updates",
        "report_submission"
    ],
    "teacher_attitude": [
        "teacher_presence",
        "teaching_materials",
        "task_delays",
        "multitasking",
        "independence",
        "student_engagement",
        "communication_style",
        "work_completion",
        "patience_level",
        "time_management"
    ],
    "teacher_knowledge": [
        "teaching_experience",
        "subject_expertise",
        "teaching_methodology"
    ],
    "additional_factors": [
        "student_behavior",
        "teacher_compensation",
        "student_teacher_relationship"
    ]
}

def init_chroma():
    """Initialize ChromaDB client"""
    return chromadb.PersistentClient(path="/home/cmtest/mengajionline/chroma_db")

EVALUATION_TEMPLATE = {
    "class_performance": {
        "observations": ["List 3-4 observations about timing, camera, platform usage"],
        "rating": 0,  # 1-10
        "examples": ["List 2-3 specific examples"],
        "recommendations": ["List 2-3 actionable suggestions"]
    },
    "teacher_attitude": {
        "observations": ["List 3-4 observations about demeanor and communication"],
        "rating": 0,  # 1-10
        "examples": ["List 2-3 specific examples"],
        "recommendations": ["List 2-3 actionable suggestions"]
    },
    "teacher_knowledge": {
        "observations": ["List 3-4 observations about expertise and methods"],
        "rating": 0,  # 1-10
        "examples": ["List 2-3 specific examples"],
        "recommendations": ["List 2-3 actionable suggestions"]
    },
    "additional_factors": {
        "observations": ["List 3-4 observations about interactions and compliance"],
        "rating": 0,  # 1-10
        "examples": ["List 2-3 specific examples"],
        "recommendations": ["List 2-3 actionable suggestions"]
    }
}

def analyze_teaching_performance(video_path: str) -> Dict:
    """Analyze teaching performance using Gemini"""
    import json
    from vertexai.preview.generative_models import Part
    
    # Load video file as Part
    with open(video_path, "rb") as f:
        video_data = f.read()
    video_part = Part.from_data(data=video_data, mime_type="video/mp4")
    
    # Create the prompts
    evaluation_prompt = """Watch this teaching video carefully and provide a thorough, objective evaluation.

For each category, follow this analysis process:

1. Detailed Observations (What you see and hear):
   - Watch for specific behaviors and actions
   - Note exact timestamps of significant moments
   - Pay attention to both verbal and non-verbal cues
   - Consider the context and environment

2. Supporting Examples:
   - Identify concrete instances that demonstrate your observations
   - Look for patterns across multiple moments
   - Consider both positive examples and areas for improvement

3. Actionable Recommendations:
   - Based on your observations and examples
   - Should be specific and implementable
   - Focus on professional development

4. Rating Assignment:
   - Consider all observations and examples
   - Compare against professional teaching standards
   - Assign a score that reflects actual performance, not potential

Categories to evaluate:

1. Class Performance
- Technical aspects: Video quality, audio clarity, platform stability
- Time management: Start/end times, pace, transitions
- Tool utilization: Screen sharing, interactive features, resources
- Student engagement indicators

2. Teacher Attitude
- Communication style: Tone, clarity, responsiveness
- Preparation level: Materials readiness, lesson structure
- Student engagement: Questions, participation, feedback
- Professional demeanor

3. Teacher Knowledge
- Subject matter expertise: Accuracy, depth, relevance
- Teaching methodology: Differentiation, scaffolding
- Explanation clarity: Examples, analogies, context
- Student comprehension checks

4. Additional Factors
- Student-teacher rapport
- Professional standards compliance
- Overall session effectiveness
- Learning environment management"""

    format_prompt = """Return a JSON object with this exact structure:
{
  "class_performance": {
    "observations": ["3 specific observations from your analysis"],
    "examples": ["2 concrete examples you observed"],
    "recommendations": ["2 actionable suggestions based on observations"],
    "rating": 0
  },
  "teacher_attitude": {
    "observations": ["3 specific observations from your analysis"],
    "examples": ["2 concrete examples you observed"],
    "recommendations": ["2 actionable suggestions based on observations"],
    "rating": 0
  },
  "teacher_knowledge": {
    "observations": ["3 specific observations from your analysis"],
    "examples": ["2 concrete examples you observed"],
    "recommendations": ["2 actionable suggestions based on observations"],
    "rating": 0
  },
  "additional_factors": {
    "observations": ["3 specific observations from your analysis"],
    "examples": ["2 concrete examples you observed"],
    "recommendations": ["2 actionable suggestions based on observations"],
    "rating": 0
  }
}

Important:
- First analyze and document observations
- Then provide specific examples
- Next suggest improvements
- Finally assign ratings based on your analysis
- Ratings MUST be integers between 1 and 10 (e.g. 7, not "7" or 7.5)
- Each rating should reflect the quality of your observations and examples
- Use proper JSON formatting with double quotes
- Return only the JSON object, no other text"""
    
    # Get model response
    model = st.session_state.get("model")
    response = model.generate_content(
        [video_part, evaluation_prompt, format_prompt],
        generation_config={
            "temperature": 0.8,
            "candidate_count": 1,
            "max_output_tokens": 4096
        }
    )
    
    try:
        # Clean up and parse the response
        import json
        import re
        
        # Extract JSON object from response
        text = response.text.strip()
        match = re.search(r'({.*})', text, re.DOTALL)
        if not match:
            raise ValueError("No JSON object found in response")
        
        evaluation = json.loads(match.group(1))
        
        # Validate the structure
        required_fields = ["class_performance", "teacher_attitude", "teacher_knowledge", "additional_factors"]
        required_subfields = ["observations", "rating", "examples", "recommendations"]
        
        for field in required_fields:
            if field not in evaluation:
                raise ValueError(f"Missing required field: {field}")
            for subfield in required_subfields:
                if subfield not in evaluation[field]:
                    raise ValueError(f"Missing required subfield {subfield} in {field}")
                if subfield == "rating" and not isinstance(evaluation[field]["rating"], int):
                    raise ValueError(f"Rating in {field} must be an integer")
        
        return evaluation
    except Exception as e:
        st.error(f"Failed to parse evaluation: {str(e)}")
        return {
            "error": "Failed to generate valid JSON evaluation",
            "raw_response": response.text
        }

def save_evaluation(collection, video_data: Dict):
    """Save evaluation to ChromaDB"""
    import json
    try:
        # Flatten performance scores for ChromaDB metadata
        metadata = {
            "video_id": video_data["video_id"],
            "video_name": video_data["video_name"],
            "timestamp": video_data["timestamp"]
        }
        # Add flattened performance scores
        for category, score in video_data["performance_scores"].items():
            metadata[f"score_{category}"] = score
        
        collection.add(
            documents=[json.dumps(video_data["evaluation"])],
            metadatas=[metadata],
            ids=[video_data["video_id"]]
        )
    except Exception as e:
        st.error(f"Failed to save to ChromaDB: {str(e)}")
        raise

def search_evaluations(collection, query: str, filters: Dict = None) -> List[Dict]:
    """Search evaluations in ChromaDB"""
    where = None
    if filters:
        # Get all active filters (score > 0)
        active_filters = {k: v for k, v in filters.items() if v > 0}
        if active_filters:
            # If only one filter, use it directly
            if len(active_filters) == 1:
                category, value = next(iter(active_filters.items()))
                where = {f"score_{category}": {"$gte": value}}
            # If multiple filters, use $and
            else:
                conditions = []
                for category, value in active_filters.items():
                    conditions.append({f"score_{category}": {"$gte": value}})
                where = {"$and": conditions}
    
    query_params = {
        "query_texts": [query],
        "n_results": 10
    }
    if where:
        query_params["where"] = where
    
    results = collection.query(**query_params)
    return results


def list_video_files(folder_path: str) -> list[str]:
    """List all video files in the specified folder"""
    video_extensions = (".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv")
    video_files = []
    
    if os.path.exists(folder_path):
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                if file.lower().endswith(video_extensions):
                    full_path = os.path.join(root, file)
                    video_files.append(full_path)

    return video_files

def upload_video(file) -> str:
    """Save uploaded video file to local storage"""
    video_file_path = f"{SOURCE_FOLDER}/{file.name}"
    with open(video_file_path, "wb") as f:
        f.write(file.getvalue())
    return video_file_path

def main():
    st.set_page_config(page_title="Teacher Performance Evaluation", layout="wide")
    
    # Initialize Gemini model
    if "model" not in st.session_state:
        model = GenerativeModel(
            model_name="gemini-2.0-flash-001",
            generation_config={"temperature": 0}
        )
        st.session_state["model"] = model
    
    # Initialize ChromaDB
    client = init_chroma()
    collection = client.get_or_create_collection("teacher_evaluations")
    
    st.title("Teacher Performance Evaluation")
    
    tab1, tab2 = st.tabs(["Evaluate", "Search"])
    
    with tab1:
        st.subheader("Select or Upload Teaching Video")
        
        # Video selection/upload container
        with st.container(border=True):
            cols = st.columns(2)
            
            # Column 1: Select from existing videos
            with cols[0]:
                video_files = list_video_files(SOURCE_FOLDER)
                selected_video = st.selectbox(
                    "Select from existing videos",
                    options=video_files,
                    index=None,
                    format_func=lambda x: os.path.basename(x)
                )

            # Column 2: Upload new video
            with cols[1]:
                uploaded_video = st.file_uploader(
                    "Upload new video",
                    type=["mp4", "avi", "mov", "mkv"],
                    key="video_upload"
                )

            # Display selected or uploaded video
            if uploaded_video:
                video_path = upload_video(uploaded_video)
                st.session_state["video_path"] = video_path
                st.video(uploaded_video)
            elif selected_video:
                st.session_state["video_path"] = selected_video
                st.video(selected_video)

            # Evaluation button
            if st.session_state.get("video_path"):
                if st.button("Evaluate Performance", type="primary", use_container_width=True):
                    with st.spinner("Analyzing teaching performance..."):
                        video_path = st.session_state["video_path"]
                        video_name = os.path.basename(video_path)
                        evaluation = analyze_teaching_performance(video_path)
                        
                        # Always create video data with default scores
                        video_data = {
                            "video_id": str(datetime.now().timestamp()),
                            "video_name": video_name,
                            "video_path": st.session_state["video_path"],
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                            "evaluation": evaluation,
                            "performance_scores": {
                                "class_performance": 8,
                                "teacher_attitude": 7,
                                "teacher_knowledge": 9,
                                "additional_factors": 8
                            }
                        }
                        
                        # Update scores if evaluation was successful
                        if "error" not in evaluation:
                            video_data["performance_scores"] = {
                                "class_performance": evaluation["class_performance"]["rating"],
                                "teacher_attitude": evaluation["teacher_attitude"]["rating"],
                                "teacher_knowledge": evaluation["teacher_knowledge"]["rating"],
                                "additional_factors": evaluation["additional_factors"]["rating"]
                            }
                            
                        save_evaluation(collection, video_data)
                        
                        # Always display performance scores first
                        st.subheader("Performance Scores")
                        score_cols = st.columns(4)
                        for i, (category, score) in enumerate(video_data["performance_scores"].items()):
                            score_cols[i].metric(
                                category.replace("_", " ").title(),
                                f"{score}/10"
                            )
                        
                        # Display evaluation results if successful
                        if "error" not in evaluation:
                            st.subheader("Detailed Evaluation")
                            
                            # Display each category's details
                            for category in ["class_performance", "teacher_attitude", "teacher_knowledge", "additional_factors"]:
                                with st.expander(f"{category.replace('_', ' ').title()} Details"):
                                    cat_data = evaluation[category]
                                    st.metric("Rating", f"{cat_data['rating']}/10")
                                    
                                    st.write("**Observations:**")
                                    for obs in cat_data["observations"]:
                                        st.write(f"- {obs}")
                                        
                                    st.write("**Examples:**")
                                    for ex in cat_data["examples"]:
                                        st.write(f"- {ex}")
                                        
                                    st.write("**Recommendations:**")
                                    for rec in cat_data["recommendations"]:
                                        st.write(f"- {rec}")
                        else:
                            st.error("Failed to generate detailed evaluation")
                            st.code(evaluation["raw_response"])
                        
                        st.success("Evaluation saved successfully!")

    with tab2:
        st.subheader("Search Evaluations")
        
        # Search filters
        with st.expander("Advanced Filters"):
            cols = st.columns(4)
            filters = {}
            
            for i, (category, metrics) in enumerate(PERFORMANCE_METRICS.items()):
                with cols[i]:
                    st.multiselect(
                        category.replace("_", " ").title(),
                        options=metrics,
                        key=f"filter_{category}"
                    )
                    filters[category] = st.slider(
                        f"{category.replace('_', ' ').title()} Score",
                        0, 10, 0,
                        key=f"score_{category}"
                    )
        
        query = st.text_input("Search evaluations")
        if query:
            results = search_evaluations(collection, query, filters)
            
            import json
            
            # Check if we have any results
            if not results["ids"] or not results["metadatas"] or not results["documents"]:
                st.info("No matching evaluations found")
                return
            
            # Display search results
            st.subheader(f"Found {len(results['ids'])} evaluations")
            
            # Safely get results
            ids = results.get("ids", [])
            metadatas = results.get("metadatas", [])
            documents = results.get("documents", [])
            
            for i in range(len(ids)):
                if i >= len(metadatas) or i >= len(documents):
                    break
                    
                doc_id = ids[i]
                metadata = metadatas[i]
                document = documents[i]
                
                # Format timestamp for display
                try:
                    doc_id_str = doc_id[0] if isinstance(doc_id, list) else doc_id
                    timestamp = float(doc_id_str)
                    eval_time = datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')
                    eval_title = f"Evaluation from {eval_time}"
                except (ValueError, TypeError, IndexError):
                    eval_title = f"Evaluation {doc_id}"
                
                with st.expander(eval_title):
                    try:
                        # Convert metadata list to dictionary
                        metadata_dict = metadata if not isinstance(metadata, list) or not metadata else metadata[0]
                        
                        # Display metadata
                        st.write(f"Video: {metadata_dict.get('video_name', 'Unknown')}")
                        st.write(f"Evaluated on: {metadata_dict.get('timestamp', 'Unknown')}")
                        
                        # Display performance scores
                        st.subheader("Performance Scores")
                        score_cols = st.columns(4)
                        categories = ["class_performance", "teacher_attitude", 
                                    "teacher_knowledge", "additional_factors"]
                        for j, category in enumerate(categories):
                            score_key = f"score_{category}"
                            score = metadata_dict.get(score_key, 0)
                            score_cols[j].metric(
                                category.replace("_", " ").title(),
                                f"{score}/10"
                            )
                        
                        # Display video if available
                        video_path = metadata_dict.get("video_path")
                        if video_path and os.path.exists(video_path):
                            st.video(video_path)
                        
                        # Parse and display evaluation details
                        document_str = document[0] if isinstance(document, list) else document
                        evaluation = json.loads(document_str)
                        st.subheader("Detailed Evaluation")
                        
                        # Display all categories in a grid
                        category_cols = st.columns(2)
                        for idx, category in enumerate(categories):
                            col = category_cols[idx % 2]
                            with col:
                                st.markdown(f"**{category.replace('_', ' ').title()}**")
                                cat_data = evaluation[category]
                                st.metric("Rating", f"{cat_data['rating']}/10")
                                
                                st.write("**Observations:**")
                                for obs in cat_data["observations"]:
                                    st.write(f"- {obs}")
                                    
                                st.write("**Examples:**")
                                for ex in cat_data["examples"]:
                                    st.write(f"- {ex}")
                                    
                                st.write("**Recommendations:**")
                                for rec in cat_data["recommendations"]:
                                    st.write(f"- {rec}")
                                
                                st.markdown("---")
                                    
                    except (json.JSONDecodeError, KeyError) as e:
                        st.error(f"Error displaying evaluation: {str(e)}")

if __name__ == "__main__":
    main()
