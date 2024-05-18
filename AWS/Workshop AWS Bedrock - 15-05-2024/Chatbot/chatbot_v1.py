import json
import boto3

bedrock = boto3.client(service_name='bedrock-runtime')

def lambda_handler(event, context):
    if (event['httpMethod'] == 'GET'):
        output = load_html()
        return {
            'statusCode': 200,
            'headers': {'Content-Type': 'text/html'},
            'body': output
        }
    elif (event['httpMethod'] == "POST"):
        prompt = event['body']
        output = chat(prompt)
        return {
            'statusCode': 200,
            'headers': {'Content-Type': 'text/html'},
            'body': output
        }
    else:
         return {
            'statusCode': 200,
            'headers': {'Content-Type': 'text/html'},
            'body': "OK"
        }

def load_html():
    html = ''
    with open('index.html', 'r') as file:
        html = file.read()
    return html

def chat(prompt):
    modelId = 'ai21.j2-ultra'
    accept = 'application/json'
    contentType = 'application/json'
    body=json.dumps({'prompt': prompt, 'maxTokens': 250, 'stopSequences': ['Human:']})
    response = bedrock.invoke_model(body=body, modelId=modelId, accept=accept,contentType=contentType)
    response_body = json.loads(response.get('body').read())
    completions = response_body['completions']
    output = ''
    for part in completions:
        output += part['data']['text']
    return output
