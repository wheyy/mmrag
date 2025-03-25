# Instructions to run

1. Set up .env file on local computer with
    
    OPENAI_API_KEY=_your openai api key_

2. After building docker image, run 

    docker run --env-file _{.env location on local machine}_ -p 8501:8501 _{your image name}_