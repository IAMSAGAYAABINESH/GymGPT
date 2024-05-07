<h1 align="center">GymGPT - A Fitness Companion</h1>
<h3 align="center">Know your workouts!</h3>

![WE_GO_JIM](https://github.com/IAMSAGAYAABINESH/GymGPT/assets/76099682/dff4b51a-5413-42cd-b55d-90b49107f9da)

----------------------------------------------------------------------------------------------------
# Description :-

Introducing GymGPT, your ultimate fitness companion powered by the cutting-edge  Mixtral-8x7B language model. 
Whether you're a seasoned gym-goer or just starting your fitness journey, GymGPT is here to revolutionize your workouts. 
With its intuitive interface and personalized recommendations, GymGPT helps you craft tailored workout plans and provides expert advice on nutrition and exercise techniques

<br>

<div align="center">
  <br>
  <video src="https://github.com/IAMSAGAYAABINESH/GymGPT/assets/76099682/0dfbd83a-9839-4520-87d1-a14b5ce46d60" width="400" />
  <br>
</div>

### Check out the live demo on Hugging Face <a href="https://huggingface.co/spaces/SagayaAbinesh/GymGpt"><img src="https://static.vecteezy.com/system/resources/previews/009/384/880/non_2x/click-here-button-clipart-design-illustration-free-png.png" width="120" height="auto"></a>

# Getting Started :

#### 1. Clone the repository:
   - ```
     git clone https://github.com/IAMSAGAYAABINESH/GymGPT.git
     ```
#### 2. Install necessary packages:
   - ```
     pip install -r requirements.txt
     ```
#### 3. Run the `ingest.py` file, preferably on kaggle or colab for faster embeddings processing and then download the `gym_vector_db` from the output folder and save it locally.
#### 4. Sign up with Together AI today and get $25 worth of free credit! 🎉 Whether you choose to use it for a short-term project or opt for a long-term commitment, Together AI offers cost-effective solutions compared to the OpenAI API. 🚀 You also have the flexibility to explore other Language Models (LLMs) or APIs if you prefer. For a comprehensive list of options, check out this link: [python.langchain.com/docs/integrations/llms](https://python.langchain.com/docs/integrations/llms) . Once signed up, seamlessly integrate Together AI into your Python environment by setting the API Key as an environment variable. 💻✨ 
   - ```
      os.environ["TOGETHER_API_KEY"] = "YOUR_TOGETHER_API_KEY"`
     ```
   - If you are going to host it in streamlit, huggingface or other...
      - Save it in the secrets variable provided by the hosting with the name `TOGETHER_API_KEY` and key as `YOUR_TOGETHER_API_KEY`.

#### 5. To run the `app.py` file, open the CMD Terminal and and type `streamlit run FULL_FILE_PATH_OF_APP.PY`.

## Contact
If you have any questions or feedback, please raise an [github issue](https://github.com/IAMSAGAYAABINESH/GymGPT/issues).

