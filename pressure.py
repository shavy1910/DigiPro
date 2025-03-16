from pynput import keyboard
import time

# Dictionary to store key press and release times
key_timing = {}
press_durations = []  # List to store the durations of key presses (to simulate pressure)

# Function to capture key press and release times
def on_press(key):
    try:
        # Record time of key press
        key_timing[key] = {'press': time.time()}
        print(f"Key {key} pressed")
    except AttributeError:
        print(f"Special key {key} pressed")

def on_release(key):
    try:
        # Calculate time difference between press and release
        press_time = key_timing[key]['press']
        release_time = time.time()
        duration = release_time - press_time
        
        # Simulate pressure based on key press duration
        pressure_simulation = duration  # Longer duration may indicate more pressure
        press_durations.append(pressure_simulation)  # Store the pressure for later analysis
        print(f"Key {key} released. Duration: {duration:.4f} seconds (Simulated Pressure: {pressure_simulation})")
        
        # If 'esc' is pressed, stop listener
        if key == keyboard.Key.esc:
            return False
    except KeyError:
        pass

# Function to start the pressure testing
def start_pressure_test():
    input("Press any key to start the pressure test. Press 'esc' when done.\n")
    
    # Set up the listener for the keyboard
    with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
        listener.join()

    # After testing is done, calculate and display the average pressure
    if press_durations:
        average_pressure = sum(press_durations) / len(press_durations)
        print(f"\nAverage pressure exerted on a key: {average_pressure:.4f} seconds")
    else:
        print("\nNo key presses recorded during the test.")

# Run the function to start the pressure testing
start_pressure_test()    