import urllib.request
import urllib.parse
import json
import io
import wave

def test_text_chat():
    print("Testing POST /chat/text...")
    url = "http://localhost:8001/chat/text"
    data = json.dumps({"text": "What is your return policy?"}).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req) as response:
            res = json.loads(response.read().decode("utf-8"))
            print("Response:", res)
            assert "response_text" in res
            print("POST /chat/text passed!\n")
    except Exception as e:
        print("POST /chat/text failed:", e)
        if hasattr(e, 'read'):
            print(e.read().decode("utf-8"))

def test_audio_chat():
    print("Testing POST /chat/audio...")
    # Create a dummy silent wav file in memory
    wav_buffer = io.BytesIO()
    with wave.open(wav_buffer, 'wb') as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(16000)
        wav_file.writeframes(b'\x00\x00' * 16000) # 1 second of silence
    
    wav_bytes = wav_buffer.getvalue()
    
    # We need to send a multipart/form-data request
    boundary = '----WebKitFormBoundary7MA4YWxkTrZu0gW'
    body = io.BytesIO()
    body.write(f'--{boundary}\r\n'.encode('utf-8'))
    body.write(b'Content-Disposition: form-data; name="audio"; filename="test.wav"\r\n')
    body.write(b'Content-Type: audio/wav\r\n\r\n')
    body.write(wav_bytes)
    body.write(f'\r\n--{boundary}--\r\n'.encode('utf-8'))
    
    req = urllib.request.Request(
        "http://localhost:8001/chat/audio",
        data=body.getvalue(),
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"}
    )
    
    try:
        with urllib.request.urlopen(req) as response:
            res = json.loads(response.read().decode("utf-8"))
            print("Audio chat success status:", res.get("success"))
            print("Transcript:", res.get("transcript"))
            print("Processing time ms:", res.get("processing_time_ms"))
            assert res.get("success") is True
            print("POST /chat/audio passed!\n")
    except Exception as e:
        print("POST /chat/audio failed:", e)
        if hasattr(e, 'read'):
            print(e.read().decode("utf-8"))

if __name__ == "__main__":
    test_text_chat()
    test_audio_chat()

