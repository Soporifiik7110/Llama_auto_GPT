from llama_cpp import Llama
from colorama import Fore
while True:
    llm = Llama(model_path="D://AI_Research//python_binary_llama.cpp//ggml-vicuna-13b-4bit-rev1.bin", n_threads=3, use_mlock=True)
    print(f"{Fore.GREEN}+++++++++++++++++++++++++++++ Your requests ++++++++++++++++++++++++++++++++\n")
    req1 = input(f"\n{Fore.WHITE}Request: ")
    req2 = input(f"\n{Fore.WHITE}Request_2:")
    output = llm(
        f"Question: {req1}?\nAnswer:", 
        max_tokens=256, #set this to 128 this a modification that i have made on the llama.py
        temperature=0.8,
        stop=["Question:", "\n"], echo=False)

    output2 = llm(
        f"Question: {req2}?\nAnswer:",
        max_tokens=256,  #set this to 128 this a modification that i have made on the llama.py
        temperature=0.8,
        stop=["Question:", "\n"], echo=False)
    
    
    print(Fore.GREEN + "\nThinking....................................................................\n")
    print(Fore.CYAN + "Running Task1.........................................")
    print(Fore.YELLOW + "++++++++++++++++++++++++++++ Task_1 +++++++++++++++++++++++++++++++++++++\n")
    agent_text = output["choices"][0]["text"]
    print(f"\n{Fore.WHITE} Task_1: \n{agent_text}")
    print(Fore.YELLOW + "\n+++++++++++++++++++++++++++++ End Task_1 +++++++++++++++++++++++++++++++++++++\n")
    print(f"{Fore.RED}++++++++++++++++++++++++++++++++Task_2 ++++++++++++++++++++++++++++++++++++++++\n")
    agent_text_2 = output2["choices"][0]["text"]
    print(f"\n{Fore.WHITE} Task_2: \n{agent_text_2}")
    print(Fore.RED + "\n+++++++++++++++++++++++++++++ End Task_2 +++++++++++++++++++++++++++++++++++++\n")
    

