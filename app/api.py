import os
import json
import re
from typing import Dict, Optional, List
from dotenv import load_dotenv

# Load environment variables
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', '.env'))
print(f"Loading environment variables:")
print(f"PROJECT_ID: {os.getenv('PROJECT_ID')}")
print(f"LOCATION: {os.getenv('LOCATION')}")
print(f"GOOGLE_APPLICATION_CREDENTIALS: {os.getenv('GOOGLE_APPLICATION_CREDENTIALS')}")
from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

class CategoryEvaluation(BaseModel):
    observations: List[str] = Field(
        description="List of specific observations from the analysis",
        example=["Clear voice projection", "Effective use of visual aids"]
    )
    examples: List[str] = Field(
        description="Concrete examples observed during evaluation",
        example=["Used interactive whiteboard at 5:30", "Engaged students with Q&A"]
    )
    recommendations: List[str] = Field(
        description="Actionable suggestions for improvement",
        example=["Incorporate more group activities", "Add regular comprehension checks"]
    )
    rating: int = Field(
        ge=1, le=10,
        description="Performance rating on a scale of 1-10",
        example=8
    )

class Evaluation(BaseModel):
    class_performance: CategoryEvaluation
    teacher_attitude: CategoryEvaluation
    teacher_knowledge: CategoryEvaluation
    additional_factors: CategoryEvaluation

class PerformanceScores(BaseModel):
    class_performance: int = Field(ge=1, le=10, example=8)
    teacher_attitude: int = Field(ge=1, le=10, example=7)
    teacher_knowledge: int = Field(ge=1, le=10, example=9)
    additional_factors: int = Field(ge=1, le=10, example=8)

class VideoAnalysisResult(BaseModel):
    video_id: str = Field(description="Unique identifier for the analysis")
    video_name: str = Field(description="Original filename of the video")
    video_path: str = Field(description="Path where the video is stored")
    timestamp: str = Field(description="UTC timestamp of the analysis")
    evaluation: Evaluation
    performance_scores: PerformanceScores

class AnalysisResponse(BaseModel):
    job_id: str = Field(description="Unique job identifier for tracking analysis progress")
    status: str = Field(description="Status of the job submission", example="accepted")

from datetime import datetime, timezone
import asyncio
from vertexai.generative_models import GenerativeModel, Part
import chromadb
import uuid
from enum import Enum

# Constants from original app
SOURCE_FOLDER = os.environ.get("SOURCE_FOLDER", os.path.abspath("app/data/video/"))
os.makedirs(SOURCE_FOLDER, exist_ok=True)

# Job status tracking
class JobStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

class StatusResponse(BaseModel):
    status: JobStatus = Field(description="Current status of the analysis job")
    error: Optional[str] = Field(None, description="Error message if the job failed")

class Job:
    def __init__(self, job_id: str, video_path: str):
        self.job_id = job_id
        self.video_path = video_path
        self.status = JobStatus.PENDING
        self.result = None
        self.error = None

# In-memory job storage (could be replaced with Redis/DB in production)
jobs = {}

description = """
# Teacher Performance Evaluation API

This API provides endpoints for analyzing teaching performance from video content using AI. 
It implements an asynchronous processing model with job status polling.

## Features

* Upload teaching videos for AI analysis
* Track analysis progress through polling
* Get detailed performance evaluations across multiple categories
* Store and retrieve evaluation history

For detailed documentation and examples, visit the /redoc endpoint.
"""

app = FastAPI(
    title="Teacher Performance Evaluation API",
    description=description,
    version="1.0.0",
    contact={
        "name": "API Support",
        "email": "support@example.com",
    },
    license_info={
        "name": "MIT",
    }
)

# Initialize Vertex AI
import vertexai
vertexai.init(
    project=os.getenv("PROJECT_ID"),
    location=os.getenv("LOCATION")
)
model = GenerativeModel(model_name="gemini-1.5-flash-002")

# Add error handling for model initialization
try:
    response = model.generate_content(
        ["Test initialization"],
        generation_config={"temperature": 0}
    )
    print("Successfully initialized Gemini model")
except Exception as e:
    print(f"Error initializing Gemini model: {str(e)}")
chroma_client = chromadb.PersistentClient(path="/home/cmtest/mengajionline/chroma_db")
collection = chroma_client.get_or_create_collection("teacher_evaluations")

async def analyze_teaching_performance(video_path: str) -> Dict:
    """Analyze teaching performance using Gemini"""
    try:
        print(f"Starting video analysis for {video_path}")
        # Load video file as Part
        with open(video_path, "rb") as f:
            video_data = f.read()
        print(f"Loaded video data: {len(video_data)} bytes")
        video_part = Part.from_data(data=video_data, mime_type="video/mp4")
        print("Created video part for Gemini")
        
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
}"""

        # Get model response
        print("Sending request to Gemini model...")
        response = model.generate_content(
            [video_part, evaluation_prompt, format_prompt],
            generation_config={
                "temperature": 0.8,
                "candidate_count": 1,
                "max_output_tokens": 4096
            }
        )
        print("Received response from Gemini")
        
        # Clean up and parse the response
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
        raise HTTPException(status_code=500, detail=str(e))

async def process_video(job: Job):
    """Process video analysis job"""
    try:
        print(f"Starting analysis for job {job.job_id}")
        job.status = JobStatus.PROCESSING
        evaluation = await analyze_teaching_performance(job.video_path)
        
        # Create video data structure
        video_data = {
            "video_id": job.job_id,
            "video_name": os.path.basename(job.video_path),
            "video_path": job.video_path,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "evaluation": evaluation,
            "performance_scores": {
                "class_performance": evaluation["class_performance"]["rating"],
                "teacher_attitude": evaluation["teacher_attitude"]["rating"],
                "teacher_knowledge": evaluation["teacher_knowledge"]["rating"],
                "additional_factors": evaluation["additional_factors"]["rating"]
            }
        }
        
        # Save to ChromaDB
        metadata = {
            "video_id": video_data["video_id"],
            "video_name": video_data["video_name"],
            "timestamp": video_data["timestamp"]
        }
        for category, score in video_data["performance_scores"].items():
            metadata[f"score_{category}"] = score
        
        collection.add(
            documents=[json.dumps(video_data["evaluation"])],
            metadatas=[metadata],
            ids=[video_data["video_id"]]
        )
        
        job.status = JobStatus.COMPLETED
        job.result = video_data
        
    except Exception as e:
        job.status = JobStatus.FAILED
        job.error = str(e)

@app.post("/analyze", 
    response_model=AnalysisResponse,
    summary="Start video analysis",
    description="Upload a video file to start the teaching performance analysis process. "
                "Returns a job ID that can be used to track the analysis progress.")
async def start_analysis(video: UploadFile):
    """Start video analysis process"""
    try:
        print(f"Received video upload: {video.filename}")
        # Generate unique job ID
        job_id = str(uuid.uuid4())
        print(f"Generated job ID: {job_id}")
        
        # Save uploaded video
        video_path = os.path.join(SOURCE_FOLDER, f"{job_id}_{video.filename}")
        with open(video_path, "wb") as f:
            f.write(await video.read())
        
        # Create and store job
        job = Job(job_id, video_path)
        jobs[job_id] = job
        
        # Start processing in background
        asyncio.create_task(process_video(job))
        
        response = {"job_id": job_id, "status": "accepted"}
        print(f"Sending response: {response}")
        return response
        
    except Exception as e:
        print(f"Error processing upload: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/status/{job_id}",
    response_model=StatusResponse,
    summary="Check analysis status",
    description="Get the current status of an analysis job using its job ID.")
async def get_status(job_id: str):
    """Get job status"""
    print(f"Checking status for job {job_id}")
    job = jobs.get(job_id)
    if not job:
        print(f"Job {job_id} not found")
        raise HTTPException(status_code=404, detail="Job not found")
    
    response = {
        "status": job.status,
        "error": job.error if job.status == JobStatus.FAILED else None
    }
    print(f"Status response for job {job_id}: {response}")
    return response

@app.get("/result/{job_id}",
    response_model=VideoAnalysisResult,
    summary="Get analysis results",
    description="Retrieve the complete analysis results for a finished job. "
                "Returns 202 if the analysis is still in progress.")
async def get_result(job_id: str):
    """Get analysis result"""
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    if job.status == JobStatus.PENDING or job.status == JobStatus.PROCESSING:
        raise HTTPException(status_code=202, detail="Analysis still in progress")
    
    if job.status == JobStatus.FAILED:
        raise HTTPException(status_code=500, detail=job.error)
    
    return job.result

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
