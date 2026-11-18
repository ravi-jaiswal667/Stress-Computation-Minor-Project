
import { useState, useRef, useEffect } from "react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer
} from "recharts";

import {
  analyzeImage,
  startSession,
  getSessionHistory,
  getSessionSummary,
  getAllSessions,
  deleteSession
} from "./services/api";
import "./App.css";

const emotions = [
  "angry", "disgust", "fearful", "happy",
  "neutral", "sad", "surprised"
];

function App() {
  const [image, setImage] = useState(null);
  const [preview, setPreview] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const [cameraActive, setCameraActive] = useState(false);
  const [graphData, setGraphData] = useState([]);

  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const streamRef = useRef(null);
  const intervalRef = useRef(null);
  const busyRef = useRef(false);
  const [sessionId, setSessionId] = useState(null);
  const [sessionSummary, setSessionSummary] = useState(null);
  const [sessionHistory, setSessionHistory] = useState([]);

  const sessionIdRef = useRef(null);
  const [previousSessions, setPreviousSessions] = useState([]);
  const [selectedSession, setSelectedSession] = useState(null);
  const [selectedSessionSummary, setSelectedSessionSummary] = useState(null);
  const [selectedSessionHistory, setSelectedSessionHistory] = useState([]);
  const [historyLoading, setHistoryLoading] = useState(false);

  // IMAGE UPLOAD
  const handleImageChange = (event) => {
    const file = event.target.files[0];

    if (!file) return;

    setImage(file);
    setPreview(URL.createObjectURL(file));
    setResult(null);
    setError("");
  };

  const handleAnalyze = async () => {
    if (!image) {
      setError("Please select an image first.");
      return;
    }

    try {
      setLoading(true);
      setError("");
      setGraphData([]);

      const data = await analyzeImage(image);
      setResult(data);

      if (data.face_detected) {
        setGraphData([{
          time: "Image",
          instantaneous: data.instantaneous_score,
          smoothed: data.experimental_score
        }]);
      }
    } catch (err) {
      setError(
        err.response?.data?.error ||
        "Unable to connect to Flask backend."
      );
    } finally {
      setLoading(false);
    }
  };

  // LOAD ALL PREVIOUS SESSIONS
  const loadPreviousSessions = async () => {
    try {
      setHistoryLoading(true);

      const data = await getAllSessions();

      setPreviousSessions(data.sessions || []);
    } catch (err) {
      setError("Unable to load previous sessions.");
    } finally {
      setHistoryLoading(false);
    }
  };

  // OPEN A PREVIOUS SESSION
  const openPreviousSession = async (id) => {
    try {
      setHistoryLoading(true);
      setSelectedSession(id);

      const [summary, history] = await Promise.all([
        getSessionSummary(id),
        getSessionHistory(id)
      ]);

      setSelectedSessionSummary(summary);
      setSelectedSessionHistory(history.readings || []);
    } catch (err) {
      setError("Unable to load the selected session.");
    } finally {
      setHistoryLoading(false);
    }
  };

  // DELETE A PREVIOUS SESSION
  const handleDeleteSession = async (id) => {
    const confirmed = window.confirm(
      "Are you sure you want to permanently delete this session?"
    );

    if (!confirmed) return;

    try {
      await deleteSession(id);

      setPreviousSessions((previous) =>
        previous.filter((session) => session.session_id !== id)
      );

      if (selectedSession === id) {
        setSelectedSession(null);
        setSelectedSessionSummary(null);
        setSelectedSessionHistory([]);
      }
    } catch (err) {
      setError("Unable to delete this session.");
    }
  };

  // WEBCAM START
  const startCamera = async () => {
    try {
      setError("");

      const stream = await navigator.mediaDevices.getUserMedia({
        video: true,
        audio: false
      });

      streamRef.current = stream;

      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
      }

      // Create a new backend session
      const session = await startSession();

      sessionIdRef.current = session.session_id;
      setSessionId(session.session_id);
      setSessionSummary(null);
      setSessionHistory([]);

      setImage(null);
      setPreview(null);
      setGraphData([]);
      setResult(null);
      setCameraActive(true);

      intervalRef.current = setInterval(captureFrame, 1000);

    } catch (err) {
      if (streamRef.current) {
        streamRef.current.getTracks().forEach(track => track.stop());
        streamRef.current = null;
      }

      setError(
        err.response?.data?.error ||
        "Unable to start webcam session."
      );
    }
  };


  const downloadSessionCSV = () => {
    if (!sessionHistory.length) {
      setError("No session readings available to download.");
      return;
    }

    const headers = [
      "Timestamp",
      "Emotion",
      "Confidence",
      "Instantaneous Score",
      "Smoothed Score",
      "Score Category",
      "Angry",
      "Disgust",
      "Fearful",
      "Happy",
      "Neutral",
      "Sad",
      "Surprised"
    ];

    const rows = sessionHistory.map((reading) => [
      reading.timestamp,
      reading.emotion,
      reading.confidence,
      reading.instantaneous_score ?? reading.experimental_score,
      reading.experimental_score,
      reading.score_category,
      reading.emotion_scores?.angry ?? 0,
      reading.emotion_scores?.disgust ?? 0,
      reading.emotion_scores?.fearful ?? 0,
      reading.emotion_scores?.happy ?? 0,
      reading.emotion_scores?.neutral ?? 0,
      reading.emotion_scores?.sad ?? 0,
      reading.emotion_scores?.surprised ?? 0
    ]);
    const csvContent = [headers, ...rows]
      .map((row) =>
        row.map((value) =>
          `"${String(value ?? "").replace(/"/g, '""')}"`
        ).join(",")
      )
      .join("\n");

    const blob = new Blob(
      ["\uFEFF", csvContent],
      { type: "text/csv;charset=utf-8;" }
    );

    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");

    link.href = url;
    link.download = `stress_session_${sessionId}.csv`;

    document.body.appendChild(link);
    link.click();
    link.remove();

    URL.revokeObjectURL(url);
  };

  // CAPTURE AND ANALYZE WEBCAM FRAME
  const captureFrame = async () => {
    if (busyRef.current) return;

    const video = videoRef.current;
    const canvas = canvasRef.current;

    if (
      !video ||
      !canvas ||
      video.readyState < 2 ||
      !video.videoWidth
    ) {
      return;
    }

    busyRef.current = true;

    try {
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;

      const context = canvas.getContext("2d");
      context.drawImage(video, 0, 0);

      const blob = await new Promise((resolve) => {
        canvas.toBlob(resolve, "image/jpeg", 0.85);
      });

      if (!blob) return;

      const data = await analyzeImage(
        blob,
        sessionIdRef.current
      );

      setResult(data);

      if (data.face_detected) {
        setGraphData((previous) => [
          ...previous.slice(-29),
          {
            time: new Date().toLocaleTimeString(),
            instantaneous: data.instantaneous_score,
            smoothed: data.experimental_score
          }
        ]);
      }
    } catch (err) {
      setError("Unable to analyze webcam frame.");
    } finally {
      busyRef.current = false;
    }
  };

  // STOP WEBCAM
  const stopCamera = async () => {
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
      intervalRef.current = null;
    }

    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop());
      streamRef.current = null;
    }

    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }

    setCameraActive(false);

    const id = sessionIdRef.current;

    if (id) {
      try {
        const [summary, history] = await Promise.all([
          getSessionSummary(id),
          getSessionHistory(id)
        ]);

        setSessionSummary(summary);
        setSessionHistory(history.readings || []);
        await loadPreviousSessions();
      } catch (err) {
        setError("Could not retrieve session report.");
      }
    }
  };

  // CLEANUP WHEN COMPONENT UNMOUNTS
  useEffect(() => {
    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
      }

      if (streamRef.current) {
        streamRef.current.getTracks().forEach(
          (track) => track.stop()
        );
      }
    };
  }, []);
  useEffect(() => {
    loadPreviousSessions();
  }, []);
  const emotionLabels = [
    "angry", "disgust", "fearful", "happy",
    "neutral", "sad", "surprised"
  ];

  const emotionDistribution = emotionLabels.map((emotion) => {
    const count = sessionHistory.filter(
      (reading) => reading.emotion === emotion
    ).length;

    return { emotion, count };
  });
  const sessionScoreHistory = sessionHistory.map((reading, index) => ({
    time: reading.timestamp
      ? new Date(reading.timestamp).toLocaleTimeString()
      : `Reading ${index + 1}`,
    instantaneous: Number(
      reading.instantaneous_score ?? reading.experimental_score ?? 0
    ),
    smoothed: Number(reading.experimental_score ?? 0)
  }));

  const getSessionDuration = () => {
    if (!sessionHistory.length) return "0 seconds";

    const start = new Date(
      sessionSummary?.started_at ?? sessionHistory[0].timestamp
    );

    const end = new Date(
      sessionHistory[sessionHistory.length - 1].timestamp
    );

    if (
      Number.isNaN(start.getTime()) ||
      Number.isNaN(end.getTime())
    ) {
      return "Unavailable";
    }

    const totalSeconds = Math.max(
      0,
      Math.floor((end - start) / 1000)
    );

    const minutes = Math.floor(totalSeconds / 60);
    const seconds = totalSeconds % 60;

    return minutes > 0
      ? `${minutes} min ${seconds} sec`
      : `${seconds} seconds`;
  };
  const confidenceValue = Math.max(
    0,
    Math.min(
      100,
      Number(
        result?.confidence ??
        result?.emotion_scores?.[result?.emotion] ??
        0
      )
    )
  );

  const scoreValue = Math.max(
    0,
    Math.min(100, Number(result?.experimental_score ?? 0))
  );
  const instantScoreValue = Math.max(
    0,
    Math.min(100, Number(result?.instantaneous_score ?? 0))
  );
  return (
    <div className="app">
      <header className="header">
        <h2>StressComputation</h2>
        <span>Facial Analysis Dashboard</span>
      </header>

      <main className="container">

        <section className="welcome">
          <h1>Facial Expression Analysis</h1>
          <p>
            Analyze facial expressions using an image
            or your webcam.
          </p>
        </section>

        {/* DASHBOARD KPI CARDS */}
        <section className="kpi-grid">

          <div className="kpi-card">
            <div className="kpi-icon blue">◎</div>
            <div>
              <p>Face Detection</p>
              <h2>
                {result
                  ? result.face_detected
                    ? "Detected"
                    : "Not Detected"
                  : "Waiting"}
              </h2>
            </div>
          </div>

          <div className="kpi-card">
            <div className="kpi-icon green">⌁</div>
            <div>
              <p>Analysis Mode</p>
              <h2>
                {cameraActive
                  ? "Live Webcam"
                  : result
                    ? "Image Analysis"
                    : "Ready"}
              </h2>
            </div>
          </div>

          <div className="kpi-card">
            <div className="kpi-icon purple">✦</div>
            <div>
              <p>Expression Classes</p>
              <h2>{emotions.length}</h2>
            </div>
          </div>

          <div className="kpi-card">
            <div className="kpi-icon orange">◷</div>
            <div>
              <p>Session Status</p>
              <h2>
                {cameraActive
                  ? "Recording"
                  : sessionSummary
                    ? "Completed"
                    : "Ready"}
              </h2>
            </div>
          </div>

        </section>

        {/* IMAGE UPLOAD */}
        <section className="upload-card">
          <h2>Image Analysis</h2>

          <input
            type="file"
            accept="image/*"
            onChange={handleImageChange}
          />

          {preview && (
            <img
              src={preview}
              alt="Selected face"
              className="preview"
            />
          )}

          <button
            onClick={handleAnalyze}
            disabled={!image || loading || cameraActive}
          >
            {loading ? "Analyzing..." : "Analyze Image"}
          </button>
        </section>

        {/* WEBCAM */}
        <section className="upload-card">
          <h2>Live Webcam Analysis</h2>

          <video
            ref={videoRef}
            autoPlay
            playsInline
            muted
            className="webcam-video"
          />

          <canvas
            ref={canvasRef}
            style={{ display: "none" }}
          />

          <div className="camera-controls">
            {!cameraActive ? (
              <button onClick={startCamera}>
                Start Webcam
              </button>
            ) : (
              <button
                className="stop-button"
                onClick={stopCamera}
              >
                Stop Webcam
              </button>
            )}
          </div>

          <p className="camera-status">
            {cameraActive
              ? "Camera is active – analyzing every second"
              : "Camera is stopped"}
          </p>
        </section>

        {error && <p className="error">{error}</p>}

        {/* RESULTS */}
        {result && (
          <>
            {!result.face_detected ? (
              <section className="result-card">
                <h2>No face detected</h2>
                <p>Please keep your face clearly visible.</p>
              </section>
            ) : (
              <>
                <section className="result-grid analysis-result-grid">

                  {/* DETECTED EMOTION */}
                  <div className="result-card emotion-highlight">
                    <p className="result-label">DETECTED EMOTION</p>

                    <div className="emotion-main">
                      <div className="emotion-symbol">☺</div>
                      <div>
                        <h1>{result.emotion}</h1>
                        <p className="result-subtitle">
                          Facial expression detected
                        </p>
                      </div>
                    </div>

                    <div className="confidence-section">
                      <div className="confidence-heading">
                        <span>Detection Confidence</span>
                        <strong>{confidenceValue.toFixed(2)}%</strong>
                      </div>

                      <div className="confidence-track">
                        <div
                          className="confidence-fill"
                          style={{ width: `${confidenceValue}%` }}
                        />
                      </div>
                    </div>
                  </div>

                  {/* EXPERIMENTAL SCORE */}
                  <div className="result-card score-highlight">
                    <p className="result-label">EXPERIMENTAL SCORE</p>

                    <div className="score-gauge">
                      <div
                        className="score-gauge-circle"
                        style={{
                          background: `conic-gradient(
            #2563eb ${scoreValue * 3.6}deg,
            #e2e8f0 ${scoreValue * 3.6}deg
          )`
                        }}
                      >
                        <div className="score-gauge-inner">
                          <strong>{scoreValue.toFixed(1)}</strong>
                          <span>/ 100</span>
                        </div>
                      </div>
                    </div>

                    <h3 className="score-category">
                      {result.score_category}
                    </h3>
                    <div className="score-comparison">
                      <p>
                        Instantaneous Score:{" "}
                        <strong>{instantScoreValue.toFixed(1)}/100</strong>
                      </p>

                      <p>
                        Smoothed Score:{" "}
                        <strong>{scoreValue.toFixed(1)}/100</strong>
                      </p>
                    </div>

                    <p className="result-subtitle">
                      Experimental expression-based score
                    </p>
                  </div>

                </section>

                {/* EMOTION PROBABILITIES */}
                <section className="result-card probability-card">
                  <div className="probability-header">
                    <div>
                      <h2>Emotion Probabilities</h2>
                      <p>Confidence distribution across seven expressions</p>
                    </div>

                    <span className="prediction-badge">
                      Predicted: {result.emotion}
                    </span>
                  </div>

                  <div className="probability-list">
                    {emotions.map((emotion) => {
                      const score = Math.max(
                        0,
                        Math.min(
                          100,
                          Number(result.emotion_scores?.[emotion] ?? 0)
                        )
                      );

                      const isPredicted = emotion === result.emotion;

                      return (
                        <div
                          className={`probability-item ${isPredicted ? "predicted-emotion" : ""
                            }`}
                          key={emotion}
                        >
                          <div className="probability-label">
                            <span>{emotion}</span>
                            <strong>{score.toFixed(2)}%</strong>
                          </div>

                          <div className="probability-track">
                            <div
                              className="probability-fill"
                              style={{ width: `${score}%` }}
                            />
                          </div>
                        </div>
                      );
                    })}
                  </div>
                </section>

                {/* LIVE GRAPH */}
                <section className="result-card">
                  <h2>Instantaneous vs Smoothed Score History</h2>

                  <div className="graph-container">
                    <ResponsiveContainer width="100%" height={300}>
                      <LineChart data={graphData}>
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis dataKey="time" />
                        <YAxis domain={[0, 100]} />
                        <Tooltip />

                        <Line
                          type="monotone"
                          dataKey="instantaneous"
                          name="Instantaneous Score"
                          stroke="#f59e0b"
                          strokeWidth={2}
                          dot={false}
                        />

                        <Line
                          type="monotone"
                          dataKey="smoothed"
                          name="Smoothed Score"
                          stroke="#2563eb"
                          strokeWidth={3}
                          dot={false}
                        />
                      </LineChart>
                    </ResponsiveContainer>
                  </div>
                </section>


                <p className="disclaimer">
                  {result.message}
                </p>
              </>
            )}
          </>
        )}

        {/* WEBCAM SESSION REPORT */}
        {sessionSummary && (
          <section className="result-card session-report">

            <div className="report-header">
              <div>
                <h2>Webcam Session Report</h2>
                <p>Your recorded facial expression analysis</p>
              </div>

              <span className="report-status">Completed</span>
            </div>

            {/* SUMMARY CARDS */}
            <div className="report-stat-grid">

              <div className="report-stat">
                <span>Total Readings</span>
                <h2>{sessionSummary.total_readings}</h2>
              </div>

              <div className="report-stat">
                <span>Average Smoothed Score</span>
                <h2>{sessionSummary.average_score}/100</h2>
              </div>

              <div className="report-stat">
                <span>Maximum Smoothed Score</span>
                <h2>{sessionSummary.maximum_score}/100</h2>
              </div>


              <div className="report-stat">
                <span>Minimum Smoothed Score</span>
                <h2>{sessionSummary.minimum_score}/100</h2>
              </div>

            </div>

            {/* SESSION INFORMATION */}
            <div className="session-information">
              <div>
                <span>Session Duration</span>
                <strong>{getSessionDuration()}</strong>
              </div>

              <div>
                <span>Latest Category</span>
                <strong>{sessionSummary.latest_category}</strong>
              </div>
            </div>

            {/* SCORE HISTORY */}
            <div className="report-chart-section">
              <h2>Instantaneous vs Smoothed Score History</h2>
              <p>Changes in experimental score during the webcam session</p>

              <div className="report-chart">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={sessionScoreHistory}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="time" />
                    <YAxis domain={[0, 100]} />
                    <Tooltip />
                    <Line
                      type="monotone"
                      dataKey="instantaneous"
                      name="Instantaneous Score"
                      stroke="#f59e0b"
                      strokeWidth={2}
                      dot={false}
                    />

                    <Line
                      type="monotone"
                      dataKey="smoothed"
                      name="Smoothed Score"
                      stroke="#2563eb"
                      strokeWidth={3}
                      dot={false}
                    />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </div>

            {/* EMOTION DISTRIBUTION */}
            <div className="report-chart-section">
              <h3>Emotion Distribution</h3>
              <p>Number of times each facial expression was detected</p>

              <div className="report-chart">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={emotionDistribution}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="emotion" />
                    <YAxis allowDecimals={false} />
                    <Tooltip />
                    <Line
                      type="monotone"
                      dataKey="count"
                      stroke="#7c3aed"
                      strokeWidth={3}
                    />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </div>

            {/* DISCLAIMER */}
            <p className="disclaimer">
              These are experimental expression-based scores,
              not validated stress measurements.
            </p>

            {/* CSV DOWNLOAD */}
            <button
              className="download-report-button"
              onClick={downloadSessionCSV}
              disabled={!sessionHistory.length}
            >
              Download Session Report (CSV)
            </button>

          </section>
        )}
        {/* PREVIOUS SESSION HISTORY */}
        <section className="result-card history-section">

          <div className="history-header">
            <div>
              <h2>Previous Sessions</h2>
              <p>Review your previously recorded webcam sessions.</p>
            </div>

            <button
              onClick={loadPreviousSessions}
              disabled={historyLoading}
            >
              {historyLoading ? "Loading..." : "Refresh"}
            </button>
          </div>

          {previousSessions.length === 0 ? (
            <p className="empty-history">
              No previous sessions found. Start a webcam session to create one.
            </p>
          ) : (
            <div className="history-list">

              {previousSessions.map((session) => (
                <div className="history-item" key={session.session_id}>

                  <div className="history-details">
                    <h3>
                      {new Date(session.started_at).toLocaleString()}
                    </h3>

                    <p>
                      {session.total_readings} readings
                    </p>

                    <span
                      className={`history-status ${session.status === "completed"
                        ? "completed"
                        : "active"
                        }`}
                    >
                      {session.status}
                    </span>
                  </div>

                  <div className="history-actions">
                    <button
                      onClick={() => openPreviousSession(session.session_id)}
                      disabled={
                        historyLoading || session.status !== "completed"
                      }
                    >
                      View Report
                    </button>

                    <button
                      className="delete-history-button"
                      onClick={() => handleDeleteSession(session.session_id)}
                      disabled={historyLoading}
                    >
                      Delete
                    </button>
                  </div>

                </div>
              ))}

            </div>
          )}

          {/* SELECTED PREVIOUS SESSION REPORT */}
          {selectedSessionSummary && (
            <div className="selected-history-report">

              <h2>Selected Session Report</h2>

              <p>
                Session ID: {selectedSession}
              </p>

              <div className="report-stat-grid">

                <div className="report-stat">
                  <span>Total Readings</span>
                  <h2>{selectedSessionSummary.total_readings}</h2>
                </div>

                <div className="report-stat">
                  <span>Average Score</span>
                  <h2>{selectedSessionSummary.average_score}</h2>
                </div>

                <div className="report-stat">
                  <span>Maximum Score</span>
                  <h2>{selectedSessionSummary.maximum_score}</h2>
                </div>

                <div className="report-stat">
                  <span>Minimum Score</span>
                  <h2>{selectedSessionSummary.minimum_score}</h2>
                </div>

              </div>

              <h3>Recorded Emotion Readings</h3>

              <div className="history-table-wrapper">
                <table className="history-table">
                  <thead>
                    <tr>
                      <th>Time</th>
                      <th>Emotion</th>
                      <th>Confidence</th>
                      <th>Instant Score</th>
                      <th>Smoothed Score</th>
                      <th>Category</th>
                    </tr>
                  </thead>

                  <tbody>
                    {selectedSessionHistory.map((reading, index) => (
                      <tr key={`${reading.timestamp}-${index}`}>
                        <td>
                          {new Date(reading.timestamp).toLocaleTimeString()}
                        </td>
                        <td className="capitalize">
                          {reading.emotion}
                        </td>
                        <td>{reading.confidence}%</td>
                        <td>
                          {reading.instantaneous_score ?? reading.experimental_score}
                        </td>
                        <td>{reading.experimental_score}</td>
                        <td>{reading.score_category}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>

              <button
                className="close-history-button"
                onClick={() => {
                  setSelectedSession(null);
                  setSelectedSessionSummary(null);
                  setSelectedSessionHistory([]);
                }}
              >
                Close Report
              </button>

            </div>
          )}

        </section>
      </main>
    </div>
  );
}

export default App;
