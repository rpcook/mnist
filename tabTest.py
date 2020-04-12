from tkinter import ttk
import tkinter as tk
#from tkinter.scrolledtext import ScrolledText


def demo():
    root = tk.Tk()
    root.title("ttk.Notebook")

    nb = ttk.Notebook(root)

    # adding Frames as pages for the ttk.Notebook 
    # first page, which would get widgets gridded into it
    page1 = ttk.Frame(nb)

    # second page
    page2 = ttk.Frame(nb)
    #text = ScrolledText(page2)
    #text.pack(expand=1, fill="both")
    w = tk.Label(page2, text="hello there second tab")
    w.pack()
	
    nb.add(page1, text='One')
    nb.add(page2, text='Two')

    nb.pack(expand=1, fill="both")

    root.mainloop()

#think this line only allows GUI if this is called directly from CMD??
#if __name__ == "__main__":
# call main GUI
demo()