import matplotlib.pyplot as plt
import numpy as np

def plot_signal_with_click(time, data, signal_name):

    fig, ax = plt.subplots()

    if isinstance(signal_name, list):
        i = 0 
        for name in signal_name:
            ax.plot(time, data[:,i], label=name)
            i += 1
    else:
        ax.plot(time, data, label=signal_name)
    
    ax.set_title('Click on the plot to to indicate the beggining of the signal')
    ax.set_xlabel('time (s)')
    ax.legend()

    # Initialize vertical line as None
    vline = None
    x = None

    def onclick(event):
        # Make vline and x accessible from outside the callback so we can modify them
        nonlocal vline
        nonlocal x
        if event.inaxes == ax:
            # get x value of clicked point
            x = event.xdata

            # Remove existing vertical line, if any
            if vline is not None:
                vline.remove()

            # Draw new vertical line
            vline = ax.axvline(x, color='red', linestyle='dotted')
            fig.canvas.draw()

    fig.canvas.mpl_connect('button_press_event', onclick)
    plt.show()
    idx = np.argmin(np.abs(time - x))
    return time[idx]

def select_range(data, signal_name):

    fig, ax = plt.subplots()

    if isinstance(signal_name, list):
        i = 0
        for name in signal_name:
            ax.plot(data[:,i], label=name)
            i += 1
    else:
        ax.plot(data, label=signal_name)

    ax.set_title('Click on the plot to select the static area (red = start, blue = end)')
    ax.set_xlabel('time (s)')
    ax.legend()

    # Initialize vertical lines and x values as None
    vline1, vline2 = None, None
    x1, x2 = None, None
    state = None

    def onclick(event):
        nonlocal vline1, vline2
        nonlocal x1, x2
        nonlocal state
        if event.inaxes == ax:
            if x1 is None:
                # get x value of first clicked point
                x1 = event.xdata

                # Remove existing vertical line, if any
                if vline1 is not None:
                    vline1.remove()

                # Draw new vertical line
                vline1 = ax.axvline(x1, color='red', linestyle='dotted')
                fig.canvas.draw()

                state = 'first'

            elif x1 is not None and x2 is None:
                # get x value of second clicked point
                x2 = event.xdata

                # Remove existing vertical line, if any
                if vline2 is not None:
                    vline2.remove()

                # Draw new vertical line
                vline2 = ax.axvline(x2, color='blue', linestyle='dotted')
                fig.canvas.draw()

                state = 'second'

            elif x1 is not None and x2 is not None:

                if state == 'second':
                    x1 = event.xdata
                    vline1.remove()
                    vline1 = ax.axvline(x1, color='red', linestyle='dotted')
                    fig.canvas.draw()
                    state = 'first'

                elif state == 'first':
                    x2 = event.xdata
                    vline2.remove()
                    vline2 = ax.axvline(x2, color='blue', linestyle='dotted')
                    fig.canvas.draw()
                    state = 'second'

    fig.canvas.mpl_connect('button_press_event', onclick)
    plt.show()

    return int(round(x1)), int(round(x2))

if __name__ == "__main__":
    # Generate some example data
    time = np.linspace(0, 10, 1000)
    data = np.sin(np.linspace(0, 10, 1000))

    # Plot the signal and wait for user click
    x1, x2 = select_range(data, 'sample_data')
    print(f'Start: {x1:.2f}')
    print(f'Stop: {x2:.2f}')