'use client';
import { useState } from 'react';

export default function ChatApp() {
  const [input, setInput] = useState('');
  const [messages, setMessages] = useState<{ role: string; content: string }[]>([]);
  const [loading, setLoading] = useState(false);

  const sendMessage = async () => {
    if (!input.trim()) return;
    
    const newMessages = [...messages, { role: 'user', content: input }];
    setMessages(newMessages);
    setInput('');
    setLoading(true);

    try {
      // Gọi API sang thẳng con FastAPI
      const res = await fetch('http://127.0.0.1:8000/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: input }),
      });

      const data = await res.json();

console.log(data);

setMessages([
  ...newMessages,
  {
    role: 'ai',
    content:
      data.ai_response ||
      data.error ||
      'Không có phản hồi từ server'
  }
]);
    } catch (error) {
      setMessages([...newMessages, { role: 'ai', content: 'Lỗi server ' }]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex flex-col h-screen bg-gray-50 text-gray-800 font-sans">
      <header className="bg-white shadow-sm p-4 text-center">
        <h1 className="text-xl font-bold text-blue-600">👨‍⚕️ Trợ lý Y Tế AIMed</h1>
      </header>
      
      <main className="flex-1 overflow-y-auto p-4 max-w-3xl mx-auto w-full space-y-4">
        {messages.length === 0 && (
          <div className="text-center text-gray-400 mt-20">Hãy đặt câu hỏi về sức khỏe của bạn...</div>
        )}
        
        {messages.map((msg, idx) => (
          <div key={idx} className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
            <div className={`p-3 max-w-[80%] rounded-xl whitespace-pre-wrap ${
              msg.role === 'user' ? 'bg-blue-600 text-white rounded-br-none' : 'bg-white border shadow-sm rounded-bl-none'
            }`}>
              {msg.content}
            </div>
          </div>
        ))}
        {loading && <div className="text-gray-400 text-sm italic pl-4">Đang chay  Neo4j...</div>}
      </main>

      <footer className="p-4 bg-white border-t max-w-3xl mx-auto w-full">
        <div className="flex gap-2">
          <input
            type="text"
            className="flex-1 border border-gray-300 p-3 rounded-lg outline-none focus:border-blue-500"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && sendMessage()}
            placeholder="Bạn đang đau ở đâu...?"
          />
          <button 
            onClick={sendMessage}
            disabled={loading}
            className="bg-blue-600 text-white px-6 py-3 rounded-lg font-bold hover:bg-blue-700 disabled:opacity-50"
          >
            Gửi
          </button>
        </div>
      </footer>
    </div>
  );
}