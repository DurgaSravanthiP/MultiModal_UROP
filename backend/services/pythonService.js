import fs from "fs";
import path from "path";
import { spawn } from "child_process";
import { fileURLToPath } from "url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const PYTHON_DIR = path.resolve(__dirname, "../../python_files");

export function runPythonPrediction(filePath, originalName = "", scanType = "lung_ct") {
  const scriptsByScanType = {
    lung_ct: "lung_ct_predict.py",
    brain_mri: "brain_mri_predict.py",
  };
  const scriptName = scriptsByScanType[scanType];
  if (!scriptName) {
    return Promise.reject(new Error(`Unsupported scan type: ${scanType}`));
  }
  const scriptPath = path.join(PYTHON_DIR, scriptName);

  return new Promise((resolve, reject) => {
    if (!fs.existsSync(scriptPath)) {
      return reject(new Error(`Python model script not found: ${scriptPath}`));
    }

    const args = [scriptPath, filePath];
    if (originalName) {
      args.push(originalName);
    }

    const child = spawn("python", args, {
      cwd: PYTHON_DIR,
      env: process.env,
    });

    let stdout = "";
    let stderr = "";

    child.stdout.on("data", (chunk) => {
      stdout += chunk.toString();
    });

    child.stderr.on("data", (chunk) => {
      stderr += chunk.toString();
    });

    child.on("error", (error) => {
      reject(new Error(`Unable to start Python prediction: ${error.message}`));
    });

    child.on("close", (code) => {
      let parsed;
      try {
        parsed = JSON.parse(stdout.trim());
      } catch (error) {
        if (code !== 0) {
          return reject(
            new Error(stderr.trim() || "Python prediction failed."),
          );
        }
        reject(
          new Error(
            `Invalid JSON returned by Python model: ${stdout.trim() || error.message}`,
          ),
        );
        return;
      }

      if (parsed.success === false) {
        return reject(new Error(parsed.message || "Prediction failed."));
      }
      if (code !== 0) {
        return reject(new Error(stderr.trim() || "Python prediction failed."));
      }
      if (!parsed.prediction || parsed.confidence === undefined) {
        return reject(new Error("Prediction output was incomplete."));
      }
      resolve(parsed);
    });
  });
}
