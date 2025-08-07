import type { Options as KyOptions } from "ky";

import NodeFormData from "form-data";
import ky from "ky";
import { Buffer } from "node:buffer";

import env from "@/env";

export async function sendTrainingRequestWithoutRetry(
  form: FormData,
  password: string,
): Promise<Response> {
  const controller = new AbortController();
  const nodeFormData = new NodeFormData();

  try {
    // Process form fields
    const fields = [
      { name: "userId", value: form.get("userId")?.toString() || "" },
      { name: "chatbotId", value: form.get("chatbotId")?.toString() || "" },
      { name: "modelType", value: form.get("modelType")?.toString() || "" },
      { name: "pdfTitle", value: form.get("pdfTitle")?.toString() || "" },
    ];

    fields.forEach(field => nodeFormData.append(field.name, field.value));

    // Process PDF file
    const pdfFile = form.get("pdf") as File;
    if (pdfFile) {
      const buffer = await pdfFile.arrayBuffer();
      nodeFormData.append("pdf", Buffer.from(buffer), {
        filename: pdfFile.name,
        contentType: pdfFile.type,
        knownLength: buffer.byteLength,
      });
    }

    // Set headers
    const headers = {
      "X-API-Password": password,
      ...nodeFormData.getHeaders(),
    };

    const response = await ky.post(`${env.PYTHON_SERVER_URL}/train/baseline-model`, {
      headers,
      body: nodeFormData.getBuffer(),
      signal: controller.signal,
      timeout: false, // Non-aktifkan timeout
      retry: 0, // Non-aktifkan retry
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error?.message || `HTTP ${response.status}`);
    }

    return response;
  }
  catch (error) {
    console.error("Training request failed:", error);
    throw error;
  }
}
