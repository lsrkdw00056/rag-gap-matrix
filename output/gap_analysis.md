# Analisis Gap

Studi terkini mengenai Retrieval-Augmented Generation (RAG) dalam domain kesehatan telah mengeksplorasi berbagai arsitektur retriever dan generator untuk mendukung pengambilan keputusan klinis. Penelitian terdahulu seperti (P04, P06, P08) menunjukkan bahwa integrasi RAG secara signifikan mampu meningkatkan akurasi faktual dibandingkan model bahasa generatif tanpa retrieval. Berbagai teknik, mulai dari pencarian padat (dense retrieval) hingga metode hybrid, telah diimplementasikan untuk menangani data medis seperti catatan keperawatan dan pedoman praktik klinis (P04, P06, P10).

Namun, analisis literatur ini mengungkapkan celah kritis pada aspek evaluasi dan metodologi. Banyak penelitian, termasuk (P01, P02, P07, P10), belum menerapkan evaluasi faithfulness yang ketat, atau hanya bergantung pada metrik kemiripan referensi yang bersifat parsial (P03, P05, P06). Selain itu, terdapat ketidakteraturan dalam laporan strategi chunking di mana sebagian studi tidak menyertakan detail segmentasi dokumen, padahal aspek ini krusial bagi kinerja model. Selain itu, masih sedikit studi yang menggabungkan evaluasi berbasis klaim dengan dataset klinis bahasa Indonesia yang autentik.

Penelitian kami bertujuan untuk mengisi kesenjangan ini dengan mengembangkan sistem RAG kesehatan yang dioptimalkan untuk bahasa Indonesia. Kami akan melakukan perbandingan sistematis terhadap berbagai strategi chunking—sesuatu yang belum dilakukan secara eksplisit pada (P01) atau (P06)—untuk mengoptimalkan konteks yang disajikan kepada LLM. Berbeda dengan studi (P06) yang hanya menggunakan metrik berbasis referensi, riset kami akan menerapkan evaluasi faithfulness berbasis verifikasi klaim terhadap dokumen sumber untuk memastikan reliabilitas klinis sistem. Fokus ini diharapkan memberikan kontribusi nyata bagi implementasi sistem pendukung keputusan medis yang akurat dan dapat dipercaya dalam ekosistem kesehatan Indonesia.

## Kandidat Gap

1. **Kurangnya Evaluasi Faithfulness yang Ketat**
   - Bukti: P01, P02, P07, dan P10 tidak melaporkan evaluasi faithfulness, sementara P03, P05, dan P06 hanya menggunakan metrik kemiripan referensi (parsial).
   - Alasan: Tanpa pengukuran langsung terhadap bukti yang diambil (grounding), sistem RAG berisiko tinggi memberikan rekomendasi medis yang menyesatkan atau berhalusinasi.
   - Usulan: Implementasi metrik evaluasi otomatis berbasis perbandingan klaim (nli-based) terhadap dokumen sumber untuk memastikan setiap pernyataan klinis didukung oleh bukti retrieved.

2. **Standardisasi Strategi Chunking**
   - Bukti: P01 dan P10 tidak melaporkan strategi chunking, sementara studi lain menggunakan metode bervariasi tanpa perbandingan empiris antar teknik.
   - Alasan: Strategi segmentasi dokumen sangat menentukan kualitas konteks yang relevan bagi LLM dalam menjawab pertanyaan klinis yang kompleks.
   - Usulan: Melakukan eksperimen sistematis membandingkan berbagai ukuran chunk (fixed vs. semantic) untuk memahami dampak efisiensi retriever terhadap akurasi jawaban klinis.

3. **Kesenjangan Bahasa dan Budaya Klinis**
   - Bukti: Mayoritas studi menggunakan bahasa Inggris (P02, P03, P04, P09); hanya P01, P06 (Indonesian) dan P10 (Chinese) yang fokus pada bahasa non-Inggris.
   - Alasan: Terminologi medis dan pedoman klinis sangat kontekstual terhadap bahasa dan sistem kesehatan lokal, sehingga model global seringkali tidak akurat di konteks Indonesia.
   - Usulan: Pengembangan framework RAG yang dioptimalkan untuk terminologi medis Indonesia dengan mengintegrasikan data pedoman praktik klinis lokal (PDS/Kemenkes).

4. **Evaluasi Berbasis Data Klinis Riil**
   - Bukti: P01 menggunakan simulasi terbatas, P07 bersifat konseptual, dan P10 menggunakan dataset kompetisi, yang semuanya kurang mencerminkan kompleksitas klinis sesungguhnya.
   - Alasan: Model yang hanya diuji pada data simulasi atau kompetisi sering kali gagal menangani ambiguitas dan kompleksitas data medis dunia nyata.
   - Usulan: Evaluasi sistem dengan melibatkan dataset klinis nyata dari fasilitas kesehatan primer melalui uji buta oleh tenaga medis profesional (expert-in-the-loop).
