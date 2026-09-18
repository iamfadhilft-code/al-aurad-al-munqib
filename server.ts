import express from 'express';
import path from 'path';
import { fileURLToPath } from 'url';
import { GoogleGenAI } from '@google/genai';
import { createServer as createViteServer } from 'vite';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();
const PORT = 3000;

app.use(express.json());

// Lazy-initialize Gemini AI client
function getGeminiClient(): GoogleGenAI | null {
  const apiKey = process.env.GEMINI_API_KEY;
  if (!apiKey) {
    console.warn('GEMINI_API_KEY is not set in environment.');
    return null;
  }
  return new GoogleGenAI({ apiKey });
}

// System instructions for Noor AI Islamic Companion
const ISLAMIC_SYSTEM_INSTRUCTION = `You are "Noor AI" (നൂർ അസിസ്റ്റന്റ് / نور المساعد), a knowledgeable, warm, and authentic Islamic spiritual companion and scholar assistant for the "AL AURAD WAL MANQIB" platform.

Core Capabilities:
1. Explain Quranic verses, authentic Hadiths, Sunnah practices, and scholarly commentaries with reverence and accuracy.
2. Provide guidance on worship, daily Adhkar (Morning/Evening litanies), Hizb, Ratib, Salawat, and Duas with Arabic text, transliteration, and Malayalam/English translations.
3. Share deep insights into Islamic history, Kerala Muslim heritage (Zainuddin Makhdoom, Ponnani, historic mosques, Mamburam Thangal, Baith, Mawlid traditions), and classical scholars.
4. If asked to find mosques, Islamic centers, Halal dining, or prayer facilities in any area (e.g. Kozhikode, Malappuram, Dubai, London, etc.), use Google Maps data to provide accurate names, addresses, neighborhood locations, and helpful details.
5. Answer gracefully in Malayalam (മലയാളം), Arabic (العربية), or English based on the user's language or request.

Tone: Respectful, uplifting, accurate, humble, and beneficial. Always include traditional Islamic etiquette (e.g. ﷺ for the Prophet Muhammad, (റ) for Sahaba).`;

// API endpoint: Multi-turn Chat with Gemini & Maps Grounding
app.post('/api/chat', async (req, res) => {
  try {
    const { messages, mode, userLocation } = req.body;

    if (!Array.isArray(messages) || messages.length === 0) {
      return res.status(400).json({ error: 'Messages array is required.' });
    }

    const ai = getGeminiClient();
    if (!ai) {
      return res.status(503).json({
        text: 'The Gemini API key is not configured in the server environment. Please set GEMINI_API_KEY in your AI Studio project settings.',
        error: 'API_KEY_MISSING',
      });
    }

    // Format conversation history for Gemini API
    const contents = messages.map((m: { role: string; content: string }) => ({
      role: m.role === 'user' ? 'user' : 'model',
      parts: [{ text: m.content }],
    }));

    // Models to try in sequence if high demand (503) or rate limits occur
    const modelsToTry = ['gemini-3.8-flash', 'gemini-flash-latest', 'gemini-3.1-flash-lite'];
    let response: any = null;
    let successfulModel = '';
    let lastError: any = null;

    // Helper to guard against hanging external calls
    const withTimeout = <T>(promise: Promise<T>, ms = 8000): Promise<T> =>
      Promise.race([
        promise,
        new Promise<T>((_, reject) => setTimeout(() => reject(new Error(`Timeout after ${ms}ms`)), ms)),
      ]);

    for (const modelToAttempt of modelsToTry) {
      try {
        if (mode === 'maps') {
          try {
            response = await withTimeout(
              ai.models.generateContent({
                model: modelToAttempt,
                contents,
                config: {
                  systemInstruction:
                    ISLAMIC_SYSTEM_INSTRUCTION +
                    (userLocation
                      ? `\nUser's current approximate location context: ${JSON.stringify(userLocation)}`
                      : ''),
                  tools: [{ googleMaps: {} }],
                },
              }),
              8000
            );
          } catch (mapsErr: any) {
            console.warn(`Maps grounding on ${modelToAttempt} notice:`, mapsErr?.message);
            response = await withTimeout(
              ai.models.generateContent({
                model: modelToAttempt,
                contents,
                config: {
                  systemInstruction: ISLAMIC_SYSTEM_INSTRUCTION,
                },
              }),
              8000
            );
          }
        } else {
          response = await withTimeout(
            ai.models.generateContent({
              model: modelToAttempt,
              contents,
              config: {
                systemInstruction: ISLAMIC_SYSTEM_INSTRUCTION,
              },
            }),
            8000
          );
        }

        if (response && response.text) {
          successfulModel = modelToAttempt;
          break;
        }
      } catch (attemptErr: any) {
        lastError = attemptErr;
        console.warn(`Model ${modelToAttempt} notice:`, attemptErr?.message || attemptErr);
      }
    }

    if (!response || !response.text) {
      return res.status(200).json({
        text:
          'The AI model servers are currently experiencing high demand. Spikes in demand are usually temporary. Please ask again in a moment, or explore our verified litanies and Quran library while the servers clear.',
        error: lastError?.message || 'High demand on upstream AI model.',
        model: 'service-advisory',
      });
    }

    const replyText = response.text || 'No response generated.';

    // Extract grounding metadata if available (places, search queries, web citations)
    const candidate = response.candidates?.[0];
    const groundingMetadata = (candidate as any)?.groundingMetadata || null;

    res.json({
      text: replyText,
      groundingMetadata,
      model: successfulModel,
    });
  } catch (error: any) {
    console.error('Gemini chat API error:', error);
    res.status(200).json({
      error: error.message || 'An error occurred while generating a response.',
      text: 'Sorry, I encountered a temporary connection issue. Please try your question again in a few moments.',
      model: 'error-advisory',
    });
  }
});

// Health check endpoint
app.get('/api/health', (req, res) => {
  res.json({
    status: 'ok',
    timestamp: new Date().toISOString(),
    geminiConfigured: Boolean(process.env.GEMINI_API_KEY),
  });
});

async function startServer() {
  if (process.env.NODE_ENV !== 'production') {
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: 'spa',
    });
    app.use(vite.middlewares);
  } else {
    const distPath = path.join(process.cwd(), 'dist');
    app.use(express.static(distPath));
    app.get('*', (req, res) => {
      res.sendFile(path.join(distPath, 'index.html'));
    });
  }

  app.listen(PORT, '0.0.0.0', () => {
    console.log(`AL AURAD WAL MANQIB Full-Stack Server running on port ${PORT}`);
  });
}

startServer();
