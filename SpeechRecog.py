import speech_recognition as sr
def record_and_transcribe():
    recognizer = sr.Recognizer()
    mic = sr.Microphone()
    print("Speak something... (recording will start in 2 seconds)")
    with mic as source:
        recognizer.adjust_for_ambient_noise(source, duration=2) 
        audio = recognizer.listen(source, timeout=5, phrase_time_limit=5)
    try:
        text = recognizer.recognize_google(audio)
        print("\n Transcription:", text)
    except sr.UnknownValueError:
        print("Could not understand audio")
    except sr.RequestError:
        print("API unavailable")
record_and_transcribe()
