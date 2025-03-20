# Replace with your actual API Key
API_KEY="YOUR_LLAMA_API_KEY"

#curl -X POST "https://api.llama.com/compat/v1/chat/completions" \
#  -H "Authorization: Bearer ${API_KEY}" \
#  -H "Content-Type: application/json" \
#  -d '{
#    "model": "llama3.3-8b-llama_api",
#    "messages": [
#      {"role": "system", "content": "You are a helpful assistant."},
#      {"role": "user", "content": "Hello!"}
#    ]
#  }'

#curl -X POST "https://api.llama.com/v1/files" \
#     -H "Content-Type: application/json" \
#     -H "Authorization: Bearer ${API_KEY}" \
#     -d '{
#           "bucket": "my_bucket",
#           "key": "uploads/seven_samurai.jpg",
#           "mime_type": "image/jpg",
#           "size": 123456
#         }'

#curl -X GET "https://api.llama.com/v1/files?bucket=my_bucket" \
#     -H "Content-Type: application/json" \
#     -H "Authorization: Bearer ${API_KEY}"

curl "https://api.llama.com/v1/models" \
  -X GET \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer ${API_KEY}"
