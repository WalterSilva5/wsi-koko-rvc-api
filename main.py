from project.core.application import Application
import uvicorn

app = Application()

if __name__ == "__main__":
    uvicorn.run("app:server", host="0.0.0.0", port=app.envs.PORT, reload=True)