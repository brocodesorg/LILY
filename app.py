"""
LILY 3D Botanical Animation Web Server (FastAPI)
Hosts the procedural 9:16 vertical animation on Render and local web browsers.

Usage:
    python app.py       # Runs locally on http://localhost:10000
On Render:
    Start Command: uvicorn app:app --host 0.0.0.0 --port $PORT
"""
import os
from pathlib import Path
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
import uvicorn

app = FastAPI(title="LILY 3D Botanical Animation")

BASE_DIR = Path(__file__).resolve().parent
VIDEO_PATH = BASE_DIR / "lily_3d_botanical_animation.mp4"

HTML_CONTENT = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🌸 LILY — 3D Botanical Animation</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            background: #06050a;
            color: #f3e8ff;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
            overflow-x: hidden;
            background-image: 
                radial-gradient(circle at 50% 30%, rgba(95, 30, 120, 0.35), transparent 65%),
                radial-gradient(circle at 80% 80%, rgba(180, 70, 160, 0.2), transparent 50%),
                radial-gradient(circle at 20% 90%, rgba(40, 20, 80, 0.4), transparent 50%);
            padding: 20px 10px;
        }

        .showcase-container {
            display: flex;
            flex-direction: column;
            align-items: center;
            gap: 16px;
            max-width: 100%;
            animation: fadeIn 0.8s cubic-bezier(0.16, 1, 0.3, 1);
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }

        .phone-mockup {
            width: 360px;
            max-width: 92vw;
            height: 640px;
            max-height: 82vh;
            background: #000;
            border-radius: 28px;
            box-shadow: 
                0 0 0 1px rgba(255, 255, 255, 0.12),
                0 24px 64px rgba(0, 0, 0, 0.85),
                0 0 40px rgba(168, 85, 247, 0.25);
            position: relative;
            overflow: hidden;
            display: flex;
            align-items: center;
            justify-content: center;
        }

        video {
            width: 100%;
            height: 100%;
            object-fit: cover;
            border-radius: 26px;
            display: block;
            cursor: pointer;
        }

        .hud-overlay {
            position: absolute;
            top: 14px;
            left: 14px;
            right: 14px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            pointer-events: none;
            z-index: 10;
        }

        .pill {
            background: rgba(15, 10, 25, 0.65);
            backdrop-filter: blur(8px);
            border: 1px solid rgba(255, 255, 255, 0.15);
            padding: 5px 12px;
            border-radius: 20px;
            font-size: 11px;
            font-weight: 500;
            letter-spacing: 0.5px;
            color: #e9d5ff;
        }

        .controls {
            display: flex;
            align-items: center;
            gap: 12px;
            background: rgba(20, 15, 30, 0.7);
            backdrop-filter: blur(12px);
            border: 1px solid rgba(255, 255, 255, 0.1);
            padding: 8px 18px;
            border-radius: 30px;
        }

        button {
            background: transparent;
            border: none;
            color: #d8b4fe;
            font-size: 13px;
            cursor: pointer;
            display: flex;
            align-items: center;
            gap: 6px;
            padding: 4px 8px;
            border-radius: 8px;
            transition: all 0.2s;
        }

        button:hover {
            color: #fff;
            background: rgba(255, 255, 255, 0.1);
        }

        .creator-tag {
            font-size: 12px;
            color: #a855f7;
            font-weight: 600;
            letter-spacing: 1.5px;
            text-transform: uppercase;
        }
    </style>
</head>
<body>

<div class="showcase-container">
    <div class="creator-tag">BROCODES • 3D BOTANICAL</div>

    <div class="phone-mockup">
        <div class="hud-overlay">
            <span class="pill">🌸 LILY • 9:16 REEL</span>
            <span class="pill">1080 × 1920</span>
        </div>

        <video id="lilyVideo" playsinline loop autoplay muted>
            <source src="/video" type="video/mp4">
            Your browser does not support the video tag.
        </video>
    </div>

    <div class="controls">
        <button id="playBtn" onclick="togglePlay()">⏸️ Pause</button>
        <button id="muteBtn" onclick="toggleMute()">🔊 Unmute</button>
        <button onclick="restartVideo()">🔄 Replay</button>
    </div>
</div>

<script>
    const video = document.getElementById('lilyVideo');
    const playBtn = document.getElementById('playBtn');
    const muteBtn = document.getElementById('muteBtn');

    function togglePlay() {
        if (video.paused) {
            video.play();
            playBtn.innerHTML = '⏸️ Pause';
        } else {
            video.pause();
            playBtn.innerHTML = '▶️ Play';
        }
    }

    function toggleMute() {
        video.muted = !video.muted;
        muteBtn.innerHTML = video.muted ? '🔊 Unmute' : '🔇 Mute';
    }

    function restartVideo() {
        video.currentTime = 0;
        video.play();
        playBtn.innerHTML = '⏸️ Pause';
    }

    video.addEventListener('click', togglePlay);
</script>

</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
async def index():
    return HTML_CONTENT


@app.get("/video")
async def get_video(request: Request):
    if not VIDEO_PATH.exists():
        raise HTTPException(status_code=404, detail="Video file not found.")

    file_size = VIDEO_PATH.stat().st_size
    range_header = request.headers.get("range")

    if range_header:
        # HTTP range request for fast video seeking/scrubbing
        range_value = range_header.strip().replace("bytes=", "")
        parts = range_value.split("-")
        start = int(parts[0]) if parts[0] else 0
        end = int(parts[1]) if len(parts) > 1 and parts[1] else file_size - 1
        content_length = (end - start) + 1

        def iter_file():
            with open(VIDEO_PATH, "rb") as f:
                f.seek(start)
                bytes_left = content_length
                chunk_size = 1024 * 512
                while bytes_left > 0:
                    read_bytes = min(chunk_size, bytes_left)
                    data = f.read(read_bytes)
                    if not data:
                        break
                    bytes_left -= len(data)
                    yield data

        headers = {
            "Content-Range": f"bytes {start}-{end}/{file_size}",
            "Accept-Ranges": "bytes",
            "Content-Length": str(content_length),
            "Content-Type": "video/mp4",
        }
        return StreamingResponse(iter_file(), status_code=206, headers=headers)

    return StreamingResponse(open(VIDEO_PATH, "rb"), media_type="video/mp4")


def main():
    port = int(os.environ.get("PORT", 10000))
    print(f"\\n🌸 LILY Web Server running on http://localhost:{port}")
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False, log_level="info")


if __name__ == "__main__":
    main()
