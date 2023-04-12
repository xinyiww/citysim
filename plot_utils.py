import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
# Define a function to be called when a point is clicked on the plot
def onclick(event, pts, xys):
    # Check if the left mouse button was clicked
    if event.button == 1:
        # Add the clicked point to the list of selected points
        x, y = event.xdata, event.ydata
        pts.append((x, y))
        # Print the selected points to the console
        print("Selected points:", pts)
        # Redraw the plot with the new point
        plt.plot(x, y, 'ro')
        
        plt.draw()
        
    # Check if the right mouse button was clicked
    elif event.button == 3:
        # Remove the most recent point from the list of selected points, if it exists
        if len(pts) > 0:
            pts.pop()
            # Clear the plot and redraw all the remaining points
            plt.clf()
            ax.axis('equal')
            plt.plot(xys[:,0], xys[:,1], c = 'grey',
            alpha = 0.3)

            for point in pts:
                plt.plot(point[0], point[1], 'ro')
            plt.draw()


def assign_points_interface(npy_fn, id, scale):
    
     
    # Create an empty list to store the selected points
    points = []
    all_lane = np.load(npy_fn, allow_pickle=True)
    # Create a Matplotlib figure and plot
    fig, ax = plt.subplots()
    # ax.axis('equal')
    xys = all_lane[id].reshape((all_lane[id].shape[0], 2))* scale
    plt.plot(xys[:,0], xys[:,1], c = 'grey',
            alpha = 0.3, label = id)
    # for (xs_p, ys_p) in traj_lst:
    #     plt.plot(xs_p, ys_p, c = 'red',
    #         alpha = 0.3)
    plt.scatter(xys[:,0], xys[:,1], color = 'grey',
            alpha = 0.3)
    plt.legend()
    # Connect the onclick function to the figure
    cid = fig.canvas.mpl_connect('button_press_event', lambda event: onclick(event, points, xys))

    # Show the plot
    plt.show()
    plt.close()
    use_assigned_points = True
    if len(points) > 0:
         
        # Convert the list of points to a NumPy array
        lcs = np.array(points)
        xs, ys = lcs[:,0], lcs[:,1]
        ds = np.cumsum(np.sqrt(np.diff(xs,prepend=xs[0])**2 + np.diff(ys,prepend=xs[0])**2))
        xs_int, ys_int = (np.interp(np.linspace(0,1,100), (ds - ds[0])/(ds[-1] - ds[0]), xs), 
                                    np.interp(np.linspace(0,1,100), (ds - ds[0])/(ds[-1] - ds[0]), ys))

        # Plot the spline
        plt.plot(xs_int, ys_int)
        plt.scatter(xs_int, ys_int)
        plt.plot(xys[:,0], xys[:,1], c = 'grey',
                alpha = 0.3)
        plt.axis("equal")
        plt.show()
    else:
        use_assigned_points = False
        xs_int = np.full((100,), np.nan)
        ys_int = np.full((100,), np.nan)
    return xs_int, ys_int, use_assigned_points

    
if __name__ == "__main__":
    # input
    scale = 0.128070 * 0.3048
    npy_fn = "dataset/McCulloch@SeminoleLanes.npy"
    assign_points_interface(npy_fn, 1, scale)
   