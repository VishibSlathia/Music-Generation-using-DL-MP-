from fastapi import FastAPI, UploadFile, File, HTTPException
import pygame
import shutil
import os

app = FastAPI()

UPLOAD_DIR = "uploaded_midi"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Initialize pygame mixer globally once
pygame.mixer.init()

@app.post("/play_midi/")
async def play_midi(file: UploadFile = File(...)):
    if not file.filename.endswith(".mid"):
        raise HTTPException(status_code=400, detail="Only MIDI (.mid) files are supported.")
    
    # Save the uploaded file
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        pygame.mixer.music.load(file_path)
        pygame.mixer.music.play()

        # Wait until playback finishes (non-blocking via loop)
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Playback failed: {str(e)}")

    return {"message": f"Successfully played {file.filename}"}
