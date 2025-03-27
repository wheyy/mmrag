# Instructions to run project

1. Set up .env file on local computer with
    
    OPENAI_API_KEY=_your openai api key_

2. Build docker image

    ```bash
    docker build -t *{image_name}* .
    ```
   
3. Run image
   
    ```bash
    docker run --env-file *{env_file}* -p 8501:8501 *{image_name}*
    ```

4. Head to http://localhost:8501/
