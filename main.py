import uvicorn
if __name__ == "__main__":
    uvicorn.run("src.app:app", host="192.168.2.173", port=8000)
