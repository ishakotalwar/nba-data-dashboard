import React from "react";
import ReactDOM from "react-dom/client";
import App from "./App";
import { initTheme } from "@/lib/theme";
import "./index.css";

// index.html already set the theme attribute before paint; this keeps the
// module's state and the DOM in agreement for the rest of the session.
initTheme();

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);
