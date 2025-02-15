# Teacher Performance Evaluation API

This API provides endpoints for analyzing teaching performance from video content using AI. It implements an asynchronous processing model with job status polling.

## API Endpoints

### Start Video Analysis

```http
POST /analyze
Content-Type: multipart/form-data
```

Upload a video file to start the analysis process.

**Request Body:**
- `video`: Video file (mp4, avi, mov, mkv)

**Response:**
```json
{
    "job_id": "uuid-string",
    "status": "accepted"
}
```

### Check Analysis Status

```http
GET /status/{job_id}
```

Check the status of an analysis job.

**Parameters:**
- `job_id`: UUID of the analysis job

**Response:**
```json
{
    "status": "string",  // pending, processing, completed, or failed
    "error": "string"    // present only if status is failed
}
```

### Get Analysis Results

```http
GET /result/{job_id}
```

Retrieve the complete analysis results.

**Parameters:**
- `job_id`: UUID of the analysis job

**Response:**
- Status 202: Analysis still in progress
- Status 404: Job not found
- Status 500: Analysis failed
- Status 200: Success, returns complete evaluation data:
```json
{
    "video_id": "string",
    "video_name": "string",
    "video_path": "string",
    "timestamp": "string",
    "evaluation": {
        "class_performance": {
            "observations": ["string"],
            "examples": ["string"],
            "recommendations": ["string"],
            "rating": 0
        },
        "teacher_attitude": {
            "observations": ["string"],
            "examples": ["string"],
            "recommendations": ["string"],
            "rating": 0
        },
        "teacher_knowledge": {
            "observations": ["string"],
            "examples": ["string"],
            "recommendations": ["string"],
            "rating": 0
        },
        "additional_factors": {
            "observations": ["string"],
            "examples": ["string"],
            "recommendations": ["string"],
            "rating": 0
        }
    },
    "performance_scores": {
        "class_performance": 0,
        "teacher_attitude": 0,
        "teacher_knowledge": 0,
        "additional_factors": 0
    }
}
```

## Usage Example

Here's how to use the API with curl:

1. Start analysis:
```bash
curl -X POST -F "video=@/path/to/video.mp4" http://localhost:8000/analyze
```

2. Check status:
```bash
curl http://localhost:8000/status/your-job-id
```

3. Get results:
```bash
curl http://localhost:8000/result/your-job-id
```

## Interactive Documentation

The API provides interactive documentation via Swagger UI at:
```
http://localhost:8000/docs
```

And ReDoc documentation at:
```
http://localhost:8000/redoc
```

## Running the API Server

Start the server with:
```bash
cd app
uvicorn api:app --reload --host 0.0.0.0 --port 8000
