from fastapi import APIRouter, UploadFile, File, HTTPException
from ..models.schemas import AudioTranscript, ErrorResponse
# from ..utils.logging import logger
import whisper
from pydub import AudioSegment
from io import BytesIO

from app.utils.logging import get_logger

# Create a logger specific to this module
logger = get_logger("app.routers.audio")


router = APIRouter(prefix="", tags=["audio"])

model = whisper.load_model("base")

def convert_to_wav(audio_bytes: bytes) -> BytesIO:
    try:
        audio = AudioSegment.from_file(BytesIO(audio_bytes))
        wav_buffer = BytesIO()
        audio.export(wav_buffer, format="wav")
        wav_buffer.seek(0)
        return wav_buffer
    except Exception as e:
        logger.error(f"Audio conversion error: {e}")
        raise HTTPException(400, "Invalid audio format")

@router.post("/audio", response_model=AudioTranscript, responses={500: {"model": ErrorResponse}})
async def transcribe_audio(file: UploadFile = File(...)):
    """Transcribe audio file to text"""
    try:
        audio_bytes = await file.read()
        wav_audio = convert_to_wav(audio_bytes)
        
        with open("temp_audio.wav", "wb") as f:
            f.write(wav_audio.read())
        
        result = model.transcribe("temp_audio.wav")
        return {"text": result.get("text", "Transcription failed")}
    
    except Exception as e:
        logger.error(f"Transcription error: {e}")
        raise HTTPException(500, "Transcription failed")