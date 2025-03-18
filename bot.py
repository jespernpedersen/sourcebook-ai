import os
import discord
from discord.ext import commands
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer, util
import torch

# Set all Hugging Face cache directories to another drive
cache_dir = os.getenv("CACHE_LOCATION")
os.environ["TRANSFORMERS_CACHE"] = cache_dir
os.environ["HF_HOME"] = cache_dir
os.environ["HUGGINGFACE_HUB_CACHE"] = cache_dir
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
print(f"Cache directory set to: {cache_dir}")

# Load environment variables
load_dotenv()
DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")

# Path to the markdown file
MARKDOWN_FILE = "phb_sourcebook.md"

# Load modal and tokenizer
try:
    model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    print("Tokenizer loaded successfully.")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        cache_dir=cache_dir,  # Explicitly set cache directory
        device_map="auto",  # Automatically use GPU if available
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,  # Use FP16 for GPU
        low_cpu_mem_usage=True,  # Optimize memory usage
    )
    # Move model to GPU if available
    if torch.cuda.is_available():
        model = model.to("cuda")
    print(f"Model device: {model.device}")
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    exit(1)

# Load a sentence transformer for semantic search
try:
    semantic_model = SentenceTransformer("all-MiniLM-L6-v2")
    print("Semantic search model loaded successfully.")
except Exception as e:
    print(f"Error loading semantic search model: {e}")
    exit(1)

# Set up the bot
intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix="/", intents=intents)

# Function to search the markdown file using semantic search
def search_markdown(query: str) -> tuple[str, str]:
    """
    Search the markdown file for the most relevant section.
    Returns a tuple of (section_header, section_content).
    """
    try:
        with open(MARKDOWN_FILE, "r", encoding="utf-8") as file:
            content = file.read()

        # Split the content into sections based on markdown headings (e.g., ## Heading)
        sections = []
        current_section = []
        current_header = ""
        for line in content.split("\n"):
            if line.startswith("##"):  # Detect markdown headings (##)
                if current_section:  # If there's content in the current section, save it
                    sections.append((current_header, "\n".join(current_section)))
                    current_section = []
                current_header = line.strip()  # Save the current header
            current_section.append(line)
        if current_section:  # Add the last section
            sections.append((current_header, "\n".join(current_section)))

        # Encode the sections and the query
        section_contents = [section[1] for section in sections]  # Use only the content for embedding
        section_embeddings = semantic_model.encode(section_contents, convert_to_tensor=True)
        query_embedding = semantic_model.encode(query, convert_to_tensor=True)

        # Compute similarity scores
        cos_scores = util.cos_sim(query_embedding, section_embeddings)[0]
        top_result = torch.argmax(cos_scores).item()

        # Get the most relevant section and its header
        relevant_header, relevant_section = sections[top_result]

        return relevant_header, relevant_section  # Return both the header and the section
    except Exception as e:
        return f"An error occurred while searching the file: {e}", ""

# Function to generate an answer using DeepSeek-R1-Distill-Qwen-1.5B
def generate_answer(query: str, context: str) -> str:
    try:
        # Construct the prompt
        prompt = (
            f"Context: {context}\n\n"
            f"Question: {query}\n\n"
            f"Using ONLY the context above, provide a concise and factual answer to the question. "
            f"Do not use any external knowledge or make assumptions. "
            f"Do not include any additional tags or markers like '</think>'. "
            f"If the question cannot be answered from the context, say 'I could not find a relevant answer in the sourcebook.'\n\n"
            f"Answer:"
        )

        # Tokenize the input
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        # Generate the response
        outputs = model.generate(
            inputs.input_ids,
            attention_mask=inputs.attention_mask,  # Explicitly set attention mask
            max_length=300,  # Keep the answer concise
            temperature=0.5,  # Lower temperature for more factual responses
            top_p=0.9,  # Use nucleus sampling for better coherence
            do_sample=True,  # Enable sampling for more diverse responses
            pad_token_id=tokenizer.eos_token_id,  # Use EOS token for padding
        )

        # Decode the generated text
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract the answer (everything after "Answer:")
        answer = generated_text.split("Answer:")[-1].strip()

        return answer
    except Exception as e:
        print(f"Error generating answer: {e}")
        return "I couldn't generate an answer. Please try again."

# Command to ask a question
@bot.command(name="ask")
async def ask(ctx, *, question: str):
    # Add a loading reaction
    loading_emoji = "⏳"  # Hourglass emoji
    await ctx.message.add_reaction(loading_emoji)

    # Search the markdown file for relevant content
    header, context = search_markdown(question)
    print(f"Header: {header}")  # Debug: Print the retrieved header
    print(f"Context: {context}")  # Debug: Print the retrieved context

    # Generate an answer using DeepSeek-R1-Distill-Qwen-1.5B
    answer = generate_answer(question, context)

    # Prepare the response
    if "could not find a relevant answer" not in answer:
        # Extract the section name from the header (e.g., "## Campaigns" -> "Campaigns")
        section_name = header.lstrip("#").strip()
        response = (
            f"{answer}\n\n"
            f"**Source:** {section_name}, Player's Handbook"
        )
    else:
        response = f"{answer}"

    # Send the response
    await ctx.send(response)

    # Remove the loading reaction and add a checkmark
    checkmark_emoji = "✅"  # Checkmark emoji
    await ctx.message.remove_reaction(loading_emoji, bot.user)
    await ctx.message.add_reaction(checkmark_emoji)

# Run the bot
bot.run(DISCORD_TOKEN)