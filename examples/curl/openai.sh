# Replace with your actual API Key
API_KEY="API_KEY"

curl -i https://api.openai.com/v1/chat/completions \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4",
    "messages": [
      {"role": "user", "content": "Hello, what is the capital of France?"}
    ],
    "temperature": 0.7
  }'


curl https://api.openai.com/v1/models \
  -H "Authorization: Bearer $API_KEY"
