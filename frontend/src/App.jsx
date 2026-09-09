import { useState, useRef } from "react";
import "./App.css";

const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

const SAMPLES = [
  { file: "normal.png", label: "Normal", type: "Fundus" },
  { file: "diabetic_retinopathy.png", label: "Diabetic retinopathy", type: "Fundus" },
  { file: "glaucoma.png", label: "Glaucoma", type: "Fundus" },
  { file: "cataract.png", label: "Cataract", type: "Fundus" },
];

const PRETTY = {
  Cataract: "Cataract",
  Diabetic_Retinopathy: "Diabetic retinopathy",
  Glaucoma: "Glaucoma",
  Keratoconus: "Keratoconus",
  Normal: "Normal",
  Uveitis: "Uveitis",
};

export default function App() {
  const [preview, setPreview] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [dragging, setDragging] = useState(false);
  const [activeSample, setActiveSample] = useState(null);
  const inputRef = useRef(null);

  async function classify(file, previewUrl, sampleFile = null) {
    setError(null);
    setResult(null);
    setPreview(previewUrl);
    setActiveSample(sampleFile);
    setLoading(true);

    const body = new FormData();
    body.append("file", file);

    try {
      const res = await fetch(`${API_URL}/predict`, { method: "POST", body });
      if (!res.ok) {
        const detail = await res.json().catch(() => null);
        throw new Error(detail?.detail || `Request failed (${res.status})`);
      }
      setResult(await res.json());
    } catch (err) {
      setError(
        err.message === "Failed to fetch"
          ? "Can't reach the classifier. Make sure the backend is running on port 8000."
          : err.message
      );
    } finally {
      setLoading(false);
    }
  }

  function handleFile(file) {
    if (!file) return;
    if (!file.type.startsWith("image/")) {
      setError("That file isn't an image. Try a JPG or PNG.");
      return;
    }
    classify(file, URL.createObjectURL(file));
  }

  async function handleSample(sample) {
    const url = `/samples/${sample.file}`;
    try {
      const res = await fetch(url);
      if (!res.ok) throw new Error("missing");
      const blob = await res.blob();
      const file = new File([blob], sample.file, { type: blob.type });
      classify(file, url, sample.file);
    } catch {
      setError(`Sample "${sample.file}" isn't in public/samples/ yet.`);
    }
  }

  const ranked = result
    ? Object.entries(result.all_scores).sort((a, b) => b[1] - a[1])
    : [];

  return (
    <div className="page">
      <header className="masthead">
        <h1>Eye disease classifier</h1>
        <p>
          An EfficientNet-B3 model sorts clinical eye images into six
          categories. Upload an image or try one of the samples.
        </p>
      </header>

      <main className="workbench">
        <section className="capture">
          <div
            className={`aperture ${dragging ? "is-dragging" : ""} ${loading ? "is-scanning" : ""}`}
            onClick={() => inputRef.current?.click()}
            onDragOver={(e) => {
              e.preventDefault();
              setDragging(true);
            }}
            onDragLeave={() => setDragging(false)}
            onDrop={(e) => {
              e.preventDefault();
              setDragging(false);
              handleFile(e.dataTransfer.files[0]);
            }}
            role="button"
            tabIndex={0}
            onKeyDown={(e) => {
              if (e.key === "Enter" || e.key === " ") {
                e.preventDefault();
                inputRef.current?.click();
              }
            }}
            aria-label="Upload an eye image"
          >
            {preview ? (
              <img src={preview} alt="Selected eye image" />
            ) : (
              <div className="aperture-empty">
                <span className="aperture-mark" aria-hidden="true" />
                <span>Drop an image here</span>
                <small>or click to browse</small>
              </div>
            )}
            <span className="scanline" aria-hidden="true" />
          </div>

          <input
            ref={inputRef}
            type="file"
            accept="image/*"
            hidden
            onChange={(e) => handleFile(e.target.files[0])}
          />

          <div className="samples">
            <p className="samples-lead">Sample images</p>
            <div className="sample-row">
              {SAMPLES.map((s) => (
                <button
                  key={s.file}
                  className={`sample ${activeSample === s.file ? "is-active" : ""}`}
                  onClick={() => handleSample(s)}
                >
                  <img src={`/samples/${s.file}`} alt="" />
                  <span className="sample-label">{s.label}</span>
                  <span className="sample-type">{s.type}</span>
                </button>
              ))}
            </div>
          </div>
        </section>

        <section className="readout" aria-live="polite">
          {error && <div className="notice notice-error">{error}</div>}

          {!error && !result && !loading && (
            <div className="notice">
              Results appear here once an image is classified.
            </div>
          )}

          {loading && <div className="notice">Analyzing image...</div>}

          {result && !loading && (
            <>
              <div className={`verdict ${result.is_disease ? "" : "is-clear"}`}>
                <span className="verdict-name">
                  {PRETTY[result.prediction] || result.prediction}
                </span>
                <span className="verdict-figure">{result.confidence}%</span>
              </div>
              <p className="verdict-note">{result.description}</p>

              <ul className="scores">
                {ranked.map(([cls, score], i) => (
                  <li key={cls} className={i === 0 ? "is-top" : ""}>
                    <span className="score-name">{PRETTY[cls] || cls}</span>
                    <span className="score-track">
                      <span
                        className="score-fill"
                        style={{ width: `${Math.max(score, 0.6)}%` }}
                      />
                    </span>
                    <span className="score-value">{score}%</span>
                  </li>
                ))}
              </ul>
            </>
          )}
        </section>
      </main>

      <footer className="footnote">
        <p>
          A demonstration project, not a diagnostic tool. The model expects
          clinical images - fundus photographs for retinal conditions, anterior
          segment photographs for corneal conditions. Photos taken on a phone
          fall outside what it was trained on, and its output for them is not
          meaningful. For anything concerning your eyes, see an optometrist.
        </p>
      </footer>
    </div>
  );
}
