import time

def calculate_typing_speed():
    sample_text = "This is just a prototype. The actual speed will be based on the daily usage of the user."
    
    print("Type the following sentence:")
    print(f"\"{sample_text}\"")
    print("Press Enter when you are ready to start typing.")
    
    input("Press Enter to start typing...")

    start_time = time.time()
    
    typed_text = input("Start typing: ")

    end_time = time.time()

    time_taken = end_time - start_time
    
    words_typed = len(typed_text) / 5

    # Calculate words per minute (WPM)
    typing_speed = (words_typed / time_taken) * 60

    # Display the results
    print(f"\nTime taken: {time_taken:.2f} seconds")
    print(f"Typing Speed: {typing_speed:.2f} words per minute (WPM)")
    
# Run the function
calculate_typing_speed()
