import { mkdir, readFile, writeFile } from "node:fs/promises";
import path from "node:path";
import { beforeAll, describe, expect, it, vi } from "vitest";

vi.setConfig({
  testTimeout: 0,
  hookTimeout: 0,
});

interface QueryItem {
  id: number;
  level?: string;
  query: string;
  ground_truth: string;
  topik?: string;
}

interface MessageResultItem {
  id: number;
  text: string;
  conversationId: number;
  userId: number;
  isBot: boolean;
  senderRole: string;
  createdAt: string;
  updatedAt: string;
  results: {
    status: string;
    query: string;
    processing_time: number;
    complexity_analysis?: Record<string, unknown>;
    search_pipeline?: Record<string, unknown>;
    results: unknown[];
    metadata: Record<string, unknown>;
    gpt_generation?: Record<string, unknown>;
    ragas_evaluation?: Record<string, unknown>;
    model_approach?: string;
    pipeline_steps?: unknown[];
    message?: string;
  };
}

describe("e2E auth, chatbot, contacts, conversations, messages", () => {
  let app: (typeof import("@/app"))["default"];

  let adminToken = "";
  let contactToken = "";

  let currentUserId = 0;

  let proposedChatbotId = 0;
  let baselineChatbotId = 0;

  let proposedConversationId = 0;
  let baselineConversationId = 0;

  let testQueries: QueryItem[] = [];

  const runId = `${Date.now()}`;
  const forwardedIp = `203.0.113.${(Date.now() % 200) + 10}`;

  const credentials = {
    name: `E2E User ${runId}`,
    email: `e2e.${runId}@example.com`,
    password: "Password123!",
  };

  const outputDir = path.resolve(process.cwd(), "tests/e2e/output");

  async function writeResultFile(fileName: string, payload: MessageResultItem[]) {
    await writeFile(path.join(outputDir, fileName), `${JSON.stringify(payload, null, 2)}\n`, "utf8");
  }

  async function readJson<T>(res: Response): Promise<T> {
    return (await res.json()) as T;
  }

  function baseHeaders() {
    return {
      "Accept": "application/json",
      "X-Forwarded-For": forwardedIp,
    };
  }

  function authHeaders(token: string) {
    return {
      ...baseHeaders(),
      Authorization: `Bearer ${token}`,
    };
  }

  beforeAll(async () => {
    app = (await import("@/app")).default;

    await mkdir(outputDir, { recursive: true });
    await writeFile(path.join(outputDir, ".gitkeep"), "", "utf8");

    const queryFilePath = path.resolve(process.cwd(), "query_test (1).json");
    const queryFile = await readFile(queryFilePath, "utf8");
    testQueries = JSON.parse(queryFile) as QueryItem[];

    expect(testQueries.length).toBeGreaterThan(0);
  });

  it("register, login, and get me", async () => {
    const registerRes = await app.request("/auth/register", {
      method: "POST",
      headers: {
        ...baseHeaders(),
        "Content-Type": "application/json",
      },
      body: JSON.stringify(credentials),
    });

    expect(registerRes.status).toBe(201);

    const loginRes = await app.request("/auth/login", {
      method: "POST",
      headers: {
        ...baseHeaders(),
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        email: credentials.email,
        password: credentials.password,
      }),
    });

    expect(loginRes.status).toBe(200);

    const loginBody = await readJson<{
      user: { id: number; email: string; name: string };
      access_token: string;
    }>(loginRes);

    expect(loginBody.user.email).toBe(credentials.email);
    expect(loginBody.access_token).toBeTypeOf("string");

    adminToken = loginBody.access_token;

    const meRes = await app.request("/auth/me", {
      method: "GET",
      headers: authHeaders(adminToken),
    });

    expect(meRes.status).toBe(200);

    const meBody = await readJson<{ id: number; email: string; name: string }>(meRes);

    expect(meBody.email).toBe(credentials.email);
    currentUserId = meBody.id;
  });

  it("create chatbot proposed model true", async () => {
    const pdfPath = path.resolve(process.cwd(), "Panduan Penggunaan Booking Engine Omni Hottilier.pdf");
    const pdfBuffer = await readFile(pdfPath);
    const pdfFile = new File([pdfBuffer], "Panduan Penggunaan Booking Engine Omni Hottilier.pdf", {
      type: "application/pdf",
    });

    const formData = new FormData();
    formData.append("title", `E2E Proposed ${runId}`);
    formData.append("description", "Chatbot e2e proposed model");
    formData.append("isPublic", "true");
    formData.append("welcomeMessage", "Selamat datang di chatbot proposed");
    formData.append("suggestionMessage", "Silakan tanyakan informasi booking engine");
    formData.append("systemPrompt", "Jawab dengan bahasa Indonesia dan ringkas.");
    formData.append("aiModel", "gpt-3.5-turbo");
    formData.append("isProposedModel", "true");
    formData.append("embeddingModel", "fasttext");
    formData.append("temperature", "0.3");
    formData.append("maxTokens", "500");
    formData.append("pdf", pdfFile);

    const createRes = await app.request("/chatbots", {
      method: "POST",
      headers: authHeaders(adminToken),
      body: formData,
    });

    expect(createRes.status).toBe(201);
  });

  it("create chatbot baseline model false", async () => {
    const pdfPath = path.resolve(process.cwd(), "Panduan Penggunaan Booking Engine Omni Hottilier.pdf");
    const pdfBuffer = await readFile(pdfPath);
    const pdfFile = new File([pdfBuffer], "Panduan Penggunaan Booking Engine Omni Hottilier.pdf", {
      type: "application/pdf",
    });

    const formData = new FormData();
    formData.append("title", `E2E Baseline ${runId}`);
    formData.append("description", "Chatbot e2e baseline model");
    formData.append("isPublic", "true");
    formData.append("welcomeMessage", "Selamat datang di chatbot baseline");
    formData.append("suggestionMessage", "Silakan tanyakan informasi booking engine");
    formData.append("systemPrompt", "Jawab dengan bahasa Indonesia dan ringkas.");
    formData.append("aiModel", "gpt-3.5-turbo");
    formData.append("isProposedModel", "false");
    formData.append("embeddingModel", "fasttext");
    formData.append("temperature", "0.3");
    formData.append("maxTokens", "500");
    formData.append("pdf", pdfFile);

    const createRes = await app.request("/chatbots", {
      method: "POST",
      headers: authHeaders(adminToken),
      body: formData,
    });

    expect(createRes.status).toBe(201);
  });

  it("get chatbots and capture proposed + baseline ids", async () => {
    const listRes = await app.request("/chatbots", {
      method: "GET",
      headers: authHeaders(adminToken),
    });

    expect(listRes.status).toBe(200);

    const listBody = await readJson<Array<{
      id: number;
      title: string;
      isProposedModel: boolean;
      isPublic: boolean;
      embeddingModel: string;
      aiModel: string;
    }>>(listRes);

    const proposed = listBody.find(item => item.title === `E2E Proposed ${runId}`);
    const baseline = listBody.find(item => item.title === `E2E Baseline ${runId}`);

    expect(proposed).toBeDefined();
    expect(baseline).toBeDefined();

    expect(proposed?.isProposedModel).toBe(true);
    expect(baseline?.isProposedModel).toBe(false);

    expect(proposed?.isPublic).toBe(true);
    expect(baseline?.isPublic).toBe(true);

    expect(proposed?.embeddingModel).toBe("fasttext");
    expect(baseline?.embeddingModel).toBe("fasttext");

    expect(proposed?.aiModel).toBe("gpt-3.5-turbo");
    expect(baseline?.aiModel).toBe("gpt-3.5-turbo");

    proposedChatbotId = proposed!.id;
    baselineChatbotId = baseline!.id;
  });

  it("get contacts and verify test@example.com belongs to current user", async () => {
    const contactsRes = await app.request("/contacts", {
      method: "GET",
      headers: authHeaders(adminToken),
    });

    expect(contactsRes.status).toBe(200);

    const contactsBody = await readJson<Array<{
      id: number;
      email: string;
      userId: number;
      name: string;
    }>>(contactsRes);

    const testContact = contactsBody.find(contact => contact.email === "test@example.com");

    expect(testContact).toBeDefined();
    expect(testContact?.userId).toBe(currentUserId);

    const contactTokenRes = await app.request("/contacts", {
      method: "POST",
      headers: {
        ...baseHeaders(),
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        name: "Test User",
        email: "test@example.com",
        chatbotId: proposedChatbotId,
      }),
    });

    expect(contactTokenRes.status).toBe(200);

    const tokenBody = await readJson<{ access_token: string }>(contactTokenRes);
    expect(tokenBody.access_token).toBeTypeOf("string");

    contactToken = tokenBody.access_token;
  });

  it("get conversations and capture ids for test user + created chatbots", async () => {
    const conversationsRes = await app.request("/conversations", {
      method: "GET",
      headers: authHeaders(adminToken),
    });

    expect(conversationsRes.status).toBe(200);

    const conversationsBody = await readJson<Array<{
      id: number;
      chatbotId: number;
      contact: {
        email: string;
        name: string;
      };
    }>>(conversationsRes);

    const proposedConversation = conversationsBody.find(item => item.chatbotId === proposedChatbotId && item.contact?.email === "test@example.com");
    const baselineConversation = conversationsBody.find(item => item.chatbotId === baselineChatbotId && item.contact?.email === "test@example.com");

    expect(proposedConversation).toBeDefined();
    expect(baselineConversation).toBeDefined();

    proposedConversationId = proposedConversation!.id;
    baselineConversationId = baselineConversation!.id;
  });

  it("send messages using all query_test data for proposed chatbot", async () => {
    const proposedResults: MessageResultItem[] = [];

    for (const item of testQueries) {
      const groundTruth = item.ground_truth.slice(0, 1000);

      const messageRes = await app.request("/messages", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          ...authHeaders(contactToken),
        },
        body: JSON.stringify({
          text: item.query,
          conversationId: proposedConversationId,
          senderRole: "admin",
          chatbotId: proposedChatbotId,
          ground_truth: groundTruth,
          results: null,
        }),
      });

      expect(messageRes.status).toBe(201);

      const messageBody = await readJson<MessageResultItem>(messageRes);

      proposedResults.push(messageBody);

      expect(messageBody.id).toBeGreaterThan(0);
      expect(messageBody.results.query).toBe(item.query);
    }

    await writeResultFile("messages-proposed-results.json", proposedResults);
  });

  it("send messages using all query_test data for baseline chatbot", async () => {
    const baselineResults: MessageResultItem[] = [];

    for (const item of testQueries) {
      const groundTruth = item.ground_truth.slice(0, 1000);

      const messageRes = await app.request("/messages", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          ...authHeaders(contactToken),
        },
        body: JSON.stringify({
          text: item.query,
          conversationId: baselineConversationId,
          senderRole: "admin",
          chatbotId: baselineChatbotId,
          ground_truth: groundTruth,
          results: null,
        }),
      });

      expect(messageRes.status).toBe(201);

      const messageBody = await readJson<MessageResultItem>(messageRes);

      baselineResults.push(messageBody);

      expect(messageBody.id).toBeGreaterThan(0);
      expect(messageBody.results.query).toBe(item.query);
    }

    await writeResultFile("messages-baseline-results.json", baselineResults);
  });
});
