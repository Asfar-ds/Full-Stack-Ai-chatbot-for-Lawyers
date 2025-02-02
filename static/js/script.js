document.addEventListener('DOMContentLoaded', () => {
    // Initialize slider values
    document.getElementById('temperature').addEventListener('input', updateTempValue);
    document.getElementById('p-value-input').addEventListener('input', updatePValue);

    // Initialize markdown parser
    marked.setOptions({
        breaks: true,
        highlight: function(code) {
            return hljs.highlightAuto(code).value;
        }
    });
});

function updateTempValue(e) {
    document.getElementById('temp-value').textContent = e.target.value;
}

function updatePValue(e) {
    document.getElementById('p-value').textContent = e.target.value;
}

async function sendMessage() {
    const input = document.getElementById('user-input');
    const message = input.value.trim();
    if (!message) return;

    addMessage(message, 'user');
    input.value = '';

    const settings = {
        model: document.getElementById('model-select').value,
        temperature: document.getElementById('temperature').value,
        p_value: document.getElementById('p-value-input').value
    };

    try {
        const response = await fetch('/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                message: message,
                ...settings
            })
        });

        const data = await response.json();
        addMessage(data.response, 'assistant');
    } catch (error) {
        console.error('Error:', error);
        addMessage('Sorry, there was an error processing your request.', 'assistant');
    }
}

async function uploadFile() {
    const fileInput = document.getElementById('file-upload');
    const file = fileInput.files[0];
    if (!file) return;

    const formData = new FormData();
    formData.append('file', file);

    try {
        const response = await fetch('/upload', {
            method: 'POST',
            body: formData
        });
        
        if (response.ok) {
            alert('File uploaded successfully!');
            fileInput.value = '';  // Clear file input
        } else {
            alert('Error uploading file');
        }
    } catch (error) {
        console.error('Error:', error);
    }
}

function addMessage(content, sender) {
    const chatHistory = document.getElementById('chat-history');
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${sender}-message`;
    
    // Sanitize and parse markdown
    const cleanContent = DOMPurify.sanitize(marked.parse(content));
    
    messageDiv.innerHTML = `
        <div class="message-header">
            <strong>${sender === 'user' ? '🙋‍♂️ You' : '🤖 Assistant'}</strong>
        </div>
        <div class="message-content">${cleanContent}</div>
    `;
    
    chatHistory.appendChild(messageDiv);
    chatHistory.scrollTop = chatHistory.scrollHeight;
}