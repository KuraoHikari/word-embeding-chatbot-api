import type { Buffer } from "node:buffer";

import { v2 as cloudinary } from "cloudinary";
import { Readable } from "node:stream";

import env from "@/env";

// Konfigurasi Cloudinary
cloudinary.config({
  cloud_name: "",
  api_key: "",
  api_secret: "",
});

// Fungsi untuk upload PDF
export async function uploadPDFToCloudinary(
  fileBuffer: Buffer,
  fileName: string,
  folder: string = "chatbot_pdfs",
): Promise<{ url: string; public_id: string }> {
  return new Promise((resolve, reject) => {
    const uploadStream = cloudinary.uploader.upload_stream(
      {
        resource_type: "raw",
        format: "pdf",
        public_id: fileName.replace(/\.[^/.]+$/, ""),
        folder,
        overwrite: false,
        type: "upload",
        allowed_formats: ["pdf"],
        tags: ["chatbot", "document"],
      },
      (error, result) => {
        if (error) {
          reject(new Error(`Gagal mengupload PDF: ${error.message}`));
        }
        else if (result) {
          resolve({
            url: result.secure_url,
            public_id: result.public_id,
          });
        }
      },
    );

    // Convert buffer ke readable stream
    const readableStream = new Readable();
    readableStream.push(fileBuffer);
    readableStream.push(null);

    readableStream.pipe(uploadStream);
  });
}
