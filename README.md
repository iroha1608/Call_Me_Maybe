*This project has been created as part of the 42 curriculum by nsato.*

<table>
	<thead>
    	<tr>
      		<th style="text-align:center">English</th>
      		<th style="text-align:center"><a href="README_ja.md">Japanese</a></th>
    	</tr>
  	</thead>
</table>

<h1>
	Call_Me_Maybe
</h1> <H2>
	Introduction to function calling in LLMs
</H2>

![image1](./.images/example_result.png)

## 📖*Content*
1. [💡Description](#1-Description)
2. [✅Instructions](#2-Instructions)
3. [⛏Additional sections](#3-Additional-sections)
	1. [Algorithm explanation](#3-1-Algorithm-explanation)
	2. [Design decisions](#3-2-Design-decisions)
	3. [Performance analysis](#3-3-Performance-analysis)
	4. [Challenges faced](#3-4-Challenges-faced)
	5. [Testing strategy](#3-5-Testing-strategy)
	6. [Example usage](#3-6-Example-usage)
4. [🎁Bonus](#4-Bonus)
	1. [Advanced Error Recovery Mechanisms](#4-1-Advanced-Error-Recovery-Mechanisms)
	2. [Performance Optimization (Caching, Batch Processing)](#4-2-Performance-Optimization-Caching-Batch-Processing)
	3. [Comprehensive Test Suite](#4-3-Comprehensive-Test-Suite)
	4. [Visualization of the Generation Process](#4-4-Visualization-of-the-Generation-Process)
	5. [Demonstration of How Encoding and Decoding Integrate with Constrained Decoding](#4-5-Demonstration-of-How-Encoding-and-Decoding-Integrate-with-Constrained-Decoding)
5. [🌈Resources](#5-Resources)
	1. [URL](#5-1-URL)
	2. [AI Usage](#5-2-AI-Usage)
  

## 💡1. Description

This project introduces function calling in Large Language Models by building a system that translates natural language prompts into structured function calls with typed arguments.  
You'll implement constrained decoding to guarantee valid JSON output, achieving near-perfect reliability with a small 0.5B parameter model, bridging the gap between human language and computer-executable operations.

## ✅2.Instructions

- If UV is not installed, please run the official UV installer script.  
```
curl -LsSf https://astral.sh/uv/install.sh | sh
```
 You can run the program by typing `make run`, but if you want to strictly follow the project PDF, please follow the steps below in order.  
*When using Makefile, only the default file can be specified.*  
  
1. Start by setting up the virtual environment.  
```
make setup
```
2. Setting up a virtual environment using UV and installing the required packages.
```
make install
```
3. **Installing required files.**  
```
make llm
```
4. Based on the PDF specifications, run the following command in the terminal to start the system.  
```
uv run python -m src [--functions_definition <function_definition_file>] [--input <input_file>] [--output <output_file>]
```
- By default, this command reads input files from the `data/input/` directory and writes the results to the `data/output/` directory.  
- The final output is a single JSON file named `data/output/function_calling_results.json`.  
- For each input natural language prompt, it generates an array of JSON objects that strictly contain the following keys. (From the PDF)  
```
prompt (string): The original natural language request
name (string): The name of the function to be called
parameters (object): Argument data that fully conforms to the schema (type)
```
5. Running unit tests.
```
make test
```
6. Code style checking and formatting.
```
make lint
```
or
```
make lint-strict
```

## 3.Additional sections

### 3-1.Algorithm explanation
- How to ensure valid JSON output  
    We use a finite state machine (FSM) to parse the generated tokens one by one, determining whether they are “part of a string,” “a key or a value,” “at what nesting depth,” and so on.  
	Based on that context, we detect tokens that violate JSON or the expected output, set the probability of those violating tokens to negative infinity, and force the model to generate correct JSON output. 

### 3-2.Design decisions
- Key Design Decisions  
  We implemented structure management using an FSM and introduced a forced queue, designing the system so that JSON structuring does not depend on the LLM’s reasoning capabilities.  
  Additionally, for numerical conversions such as `twenty-five -> 25`, we did not require an exact match with the prompt in order to leverage the LLM’s reasoning capabilities.  
	However, to prevent unnecessary sequences of zeros and infinite loops, we implemented safeguards such as character limits for each field.  

### 3-3.Performance analysis
- Accuracy, Speed, and Reliability of the Solution  
    Although this is a lightweight model with 0.5 billion parameters, it reliably generates JSON output with nearly 100% accuracy, even in edge cases.  
    Furthermore, by confining the LLM’s inference to within the parameters and managing mandatory syntax elements via a forced queue, we maximize the LLM’s inference capabilities.  

### 3-4.Challenges faced
- Since the `type` specified in the assignment was `number` rather than `int` or `float`, the required output format was unclear, and I was unable to handle it. I decided to leave this aspect to the LLM's inference.  
- When switching from JSON structuring to management via a forced queue, I worried whether this would amount to hard-coding. However, since the specified items—such as “name,” “parameters,” and “type”—must be retrieved regardless, I determined that the priority here was understanding the coding constraints. Given the challenges of retrieving the aforementioned values, I concluded it would be best to fully leverage the LLM’s reasoning capabilities and implemented this approach.  
- Regarding the handling of ‘\“’, there were instances where, if the prompt contained ‘\”’, the escape sequence was consumed when passed to the LLM, causing it to output double quotes and breaking the FSM. To address this, I implemented a fix where, if the prompt contains ‘\"’, it is escaped before being passed to the LLM.  
- During implementation, we encountered an infinite loop where the output would repeat “0000...” when generating numerical values. This occurred because the tokenizer was treating line breaks as “Ċ” when adding spaces and line breaks to the trie tree, preventing them from being properly added to the cache.  

### 3-5.Testing strategy
- How the implementation was verified  
    We divided the cases into normal cases and edge cases (empty strings, special characters, very long numbers, and handling of null and boolean values) and ran tests for each.　　
- As a bonus, we made debugging easier by visualizing the token generation process.  
- Also, as another bonus, we built a set of unit tests using pytest and unittest.mock. By mocking external I/O and LLM inference, we verified the accuracy of the implementation.  

### 3-6.Example usage

`make run`  
or  
`uv run python3 -m src -f <function file path> -i <prompt file path> -o <output file>`  
to execute the specified file.  


- Specify function definitions as follows. 

![image2](./.images/example_function.png)  

- Similarly, specify the prompt as follows.  

![image3](./.images/example_prompt.png)  

- The output file will look like this.  

![image4](./.images/example_output.png)

## 🎁4. Bonus

- We implemented the following five features:
    - Advanced error recovery mechanism
    - Performance optimization (caching, batch processing)
    - Comprehensive test suite
    - Visualization of the generation process
    - Demonstration of how to integrate encoding and decoding with constrained decoding
	
### 4-1. Advanced Error Recovery Mechanisms
- A mechanism to fall back to the original logit when filtering using ConstraintFilter crashes (when catching an `except` or when no permission token is present)  
- A design that prevents the entire system from crashing even if an error occurs.  
- Creation of custom errors. Easy-to-understand operation showing where the error occurred (engine, llm_client, tokenizer).  

### 4-2. Performance Optimization (Caching, Batch Processing)
- Caching and resumption during current text analysis.  
- Caching using Trie trees.  
- Regarding batch processing, it does not work well with FSMs. Additionally, since the PCs in the school building environment cannot use GPUs and are already utilizing the CPU to its full capacity, we decided to skip this for now.  

### 4-3. Comprehensive Test Suite

- Built a set of unit tests (test/) using pytest and unittest.mock to mock external I/O and LLM inference.

### 4-4. Visualization of the Generation Process

- Visualization of tokens before and after applying constraints to token logits, along with displaying the currently generated string.
- Screen rendering controlled using escape sequences.

### 4-5. Demonstration of How Encoding and Decoding Integrate with Constrained Decoding
- Implement an autoregressive loop in src/engine.py to demonstrate how these components work together as a pipeline.
- Within ConstraintsFilter, decode a specific schema when using a forced queue.

## 🌈5. Resources

### 5-1. URL

- [組み込み例外](https://docs.python.org/ja/3.14/library/exceptions.html)
- [Pythonのraiseについて](https://zenn.dev/tektek/articles/9b8fd47e2cac4f)
- [Function Callingとは？仕組みや使い方をわかりやすく解説](https://aismiley.co.jp/ai_news/what-is-function-calling/)
- [生成AIアプリをより多機能に(Function Calling)](https://qiita.com/ksonoda/items/1ba3916c9ee9f4d9c10c)
- [ArgumentParserの使い方](https://qiita.com/kzkadc/items/e4fc7bc9c003de1eb6d0)
- [＃９🤖Constrained Decoding(制約付きデコーディング)【オルティス教授の松尾研LLMコンペ単語帳】](https://note.com/ortiz_aipartners/n/nbd1effbf4129)
- [すごいTrie](https://qiita.com/minaminao/items/caf6d8147c7e70b6ae63)
- [プッシュダウン・オートマトン](https://ja.wikipedia.org/wiki/%E3%83%97%E3%83%83%E3%82%B7%E3%83%A5%E3%83%80%E3%82%A6%E3%83%B3%E3%83%BB%E3%82%AA%E3%83%BC%E3%83%88%E3%83%9E%E3%83%88%E3%83%B3)
- [LLM構造化出力とConstrained Decoding実践ガイド: JSON Schemaから本番アプリケーションまで](https://www.youngju.dev/blog/llm/2026-03-07-llm-structured-output-constrained-decoding-json-schema.ja)
- [自作トークナイザーを作ってみた。](https://tech-blog.cloud-config.jp/2024-12-25-tokenizer-from-scratch)
- [Qwen 3](https://note.com/npaka/n/n43abd5843fe7)
- [pyTorchのTensor型とは](https://qiita.com/mathlive/items/241bfb42d852bb801b96)
- [イラストで理解するSOLID原則](https://qiita.com/baby-degu/items/d058a62f145235a0f007)
- [ANSIエスケープシーケンス チートシート](https://qiita.com/PruneMazui/items/8a023347772620025ad6)
- [Deep Learning with Python](https://deeplearningwithpython.io/chapters/)
- [Dive into Deep Learning](https://d2l.ai/)

### 5-2. AI Usage

- Consultation on creating a script to retrieve the latest 300 review entries from the 42 project page.  
-> Creating a Markdown file using Gemini CLI based on that script.  
- Brainstorming solutions regarding whether to adapt the output format for constrained decoding to match the assignment requirements.  
- Brainstorming edge cases.  
- Consultation on code refactoring.
- Summarizing my contributions to the creation of README.md. Creating a table of contents.
- Translation (using DeepL) for the English version of README.md