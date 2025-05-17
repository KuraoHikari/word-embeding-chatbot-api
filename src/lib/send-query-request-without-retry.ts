import type { Options as KyOptions } from "ky";

import NodeFormData from "form-data";
import ky from "ky";

import env from "@/env";

interface QueryParams {
  query: string;
  userId: string;
  chatbotId: string;
  modelType: string;
  pdfTitle: string;
  topK?: number;
  similarityThreshold?: number;
}

export async function sendQueryRequestWithoutRetry(
  params: QueryParams,
  password: string,
): Promise<Response> {
  const controller = new AbortController();
  const nodeFormData = new NodeFormData();

  try {
    // Process form fields
    const fields = [
      { name: "query", value: params.query },
      { name: "userId", value: params.userId },
      { name: "chatbotId", value: params.chatbotId },
      { name: "modelType", value: params.modelType },
      { name: "pdfTitle", value: params.pdfTitle },
      { name: "topK", value: params.topK?.toString() || "5" },
      { name: "similarityThreshold", value: params.similarityThreshold?.toString() || "0.4" },
    ];

    fields.forEach(field => nodeFormData.append(field.name, field.value));

    // Set headers
    const headers = {
      "X-API-Password": password,
      ...nodeFormData.getHeaders(),
    };

    const response = await ky.post(`${env.PYTHON_SERVER_URL}/query`, {
      headers,
      body: nodeFormData.getBuffer(),
      signal: controller.signal,
      timeout: false, // Disable timeout
      retry: 0, // Disable retry
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error?.message || `HTTP ${response.status}`);
    }

    return response;
  }
  catch (error) {
    console.error("Query request failed:", error);
    throw error;
  }
}
