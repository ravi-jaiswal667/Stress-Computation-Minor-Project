
import axios from "axios";

const API = axios.create({
    baseURL: "http://127.0.0.1:5000/api",
});

// Analyze image or webcam frame
export const analyzeImage = async (image, sessionId = null) => {
    const formData = new FormData();
    formData.append("frame", image);

    if (sessionId) {
        formData.append("session_id", sessionId);
    }

    const response = await API.post("/analyze", formData);
    return response.data;
};

// Start a new webcam session
export const startSession = async () => {
    const response = await API.post("/session/start");
    return response.data;
};

// Get session history
export const getSessionHistory = async (sessionId) => {
    const response = await API.get(
        `/session/${sessionId}/history`
    );
    return response.data;
};

// Get session summary
export const getSessionSummary = async (sessionId) => {
    const response = await API.get(
        `/session/${sessionId}/summary`
    );
    return response.data;
};


export const getAllSessions = async () => {
    const response = await API.get("/sessions");
    return response.data;
};

export const deleteSession = async (sessionId) => {
    const response = await API.delete(`/session/${sessionId}`);
    return response.data;
};