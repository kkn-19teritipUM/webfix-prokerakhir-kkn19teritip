const express = require('express');
const bodyParser = require('body-parser');
const fs = require('fs');
const pdfParse = require('pdf-parse');
const path = require('path');
const { ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings } = require('@langchain/google-genai');
const { RecursiveCharacterTextSplitter } = require('langchain/text_splitter');
const { MemoryVectorStore } = require('langchain/vectorstores/memory');
const { RetrievalQAChain } = require('langchain/chains');
require('dotenv').config();

const app = express();
const PORT = process.env.PORT || 3000;

// Middlewares
app.use(bodyParser.json());
app.use(express.static(path.join(__dirname, 'public')));
app.set('view engine', 'ejs');
app.set('views', path.join(__dirname, 'views'));

// Set API Key
const GOOGLE_API_KEY = process.env.GOOGLE_API_KEY;
if (!GOOGLE_API_KEY) {
  throw new Error("GOOGLE_API_KEY tidak ditemukan di .env");
}
// process.env.GOOGLE_API_KEY = GOOGLE_API_KEY;

// Global variables
let qaChain;

// Load and prepare the PDF
async function initPDFQA() {
  const pdfBuffer = fs.readFileSync(path.join(__dirname, 'Binjai.pdf'));
  const pdfData = await pdfParse(pdfBuffer);

  const allText = pdfData.text;
  const splitter = new RecursiveCharacterTextSplitter({
    chunkSize: 1000,
    chunkOverlap: 100,
  });

  const docs = await splitter.createDocuments([allText]);

  const embeddings = new GoogleGenerativeAIEmbeddings({
    model: 'models/embedding-001',
  });

  const vectorStore = await MemoryVectorStore.fromDocuments(docs, embeddings);

  const retriever = vectorStore.asRetriever();

  const llm = new ChatGoogleGenerativeAI({
    model: 'gemini-1.5-flash',
    maxOutputTokens: 500,
  });

  qaChain = RetrievalQAChain.fromLLM(llm, retriever);
}

// Middleware untuk membatasi hanya Android & iOS
// app.use((req, res, next) => {
//   const userAgent = req.headers['user-agent'] || '';

//   const isMobile = /android|iphone|ipad|ipod/i.test(userAgent);

//   if (!isMobile) {
//     return res.status(403).send('Akses hanya diperbolehkan dari perangkat Android/iOS.');
//   }

//   next();
// });

// Routes
app.get('/', (req, res) => {
  res.render('kkn1');
});

app.get('/kkn1', (req, res) => {
  res.render('kkn1');
});

app.get('/kkn2', (req, res) => {
  res.render('kkn2');
});

app.get('/kkn3', (req, res) => {
  res.render('kkn3');
});

app.get('/kkn4', (req, res) => {
  res.render('kkn4');
});

app.get('/kkn5', (req, res) => {
  res.render('kkn5');
});

app.post('/process', async (req, res) => {
  const { topic } = req.body;

  if (!topic) {
    return res.json({ output: 'Topik kosong, silakan isi pertanyaan.' });
  }

  try {
    const response = await qaChain.call({ query: topic });
    return res.json({ output: response.text.trim() });
  } catch (error) {
    console.error('Error processing question:', error);
    return res.status(500).json({ output: 'Terjadi kesalahan dalam memproses pertanyaan.' });
  }
});

// Start server
initPDFQA().then(() => {

app.listen(PORT, '0.0.0.0', () => {
  console.log(`Server is running on http://localhost:${PORT}`);
});

}).catch(err => {
  console.error('Failed to initialize PDF QA:', err);
});



