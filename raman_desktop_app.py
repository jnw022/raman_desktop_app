# raman_desktop_app.py

import os.path
import traceback
import pickle
import json
from datetime import datetime

import PySimpleGUI as sg
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from pyspectra.readers.read_spc import read_spc
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression

from utils import MSC, Product_Classifier, PLS_Model, Model_Container

matplotlib.use('TKAgg')


#-------------------------------------------------------------------------------------------------
# global variables:
#-------------------------------------------------------------------------------------------------
figure_agg = None    # figure canvas for plotting spectra
folder = None        # selected folder containing spectral files
analyst_mode = True  # toggle for user mode, i.e. whether to show only pass/fail or detailed output
descending = True    # sorting order for files 
filename = None      # name of selected file
model = None         # selected model file



def draw_figure(canvas, figure):
    """Helper function for drawing with TKAgg backend"""
    figure_agg = FigureCanvasTkAgg(figure, canvas)
    figure_agg.draw()
    figure_agg.get_tk_widget().pack(side='top', fill='both', expand=1)
    return figure_agg

def convert_spc(spectrum, figsize=(8,4)):
    """
    Converts an .spc file into a pyplot figure that can be drawn to the screen.
    
    PARAMETERS:
    spectrum:  (string)  Path to an .spc file.
    figsize:   (tuple)   Pair of integers for the width and height of the pyplot figure.

    RETURN VALUES:
    plt.gcf()  Figure object containing the plotted spectrum.
    spec       (ndarray) Numpy array representing the spectrum.
    """

    # load the .spc file
    spec = read_spc(spectrum)
    title = os.path.basename(spectrum)

    # make the plot
    plt.figure(figsize=figsize)
    plt.plot(spec)
    plt.xlabel('Frequency Shift (cm^-1)')
    plt.ylabel('Counts')
    plt.title(title)

    return plt.gcf(), spec

def load_model(model_file):
    """Load in a pickle file containing the chemometric model."""

    if model_file.endswith('.pkl'):
        try:
            with open(model_file, 'rb') as pickle_file:
                model = pickle.load(pickle_file)  

        except Exception as e:
            tb = traceback.format_exc()
            sg.Popup("ERROR", "An exception has occurred while loading a model file!", e, tb, keep_on_top=True)

    else:
        sg.Popup('ERROR', 'Please select a Python pickle file (.pkl).', keep_on_top=True)
        model = None

    return model

def pull_folder(folder, sort_by_date=True, descending=True):
    """Returns a list of all the filenames of spectral files in a given folder and their modified dates.
       If the sort_by_date parameter is True, then output will be sorted by modified date.  Otherwise, the
       output is sorted alphabetically.  
       If the descending parameter is True, sorting will be done in descending order.  Ascending otherwise.
    """
    
    try:
        # get list of files in folder
        file_list = os.scandir(folder)
    except:
        file_list=[]

    # bit of a nasty list comprehension.
    # the idea is to collect a list of tuples of file names and their modified dates.
    # e.g. ('8675309.spc', '10 Jan 2022 09:21:37')
    filenames = [
        (file.name, datetime.fromtimestamp(file.stat().st_mtime).strftime("%d %B %Y %I:%M:%S"))
        for file in file_list
        if os.path.isfile(os.path.join(folder, file))
        and file.name.lower().endswith(".spc")
    ]
    
    # sort by modified date, if desired.
    if sort_by_date:  
        # this sorting works by turning the date string back into a datetime object, 
        # which can be readily compared to other datetime objects.
        filenames.sort(reverse=descending, key=lambda x: datetime.strptime(x[1], "%d %B %Y %I:%M:%S"))
        
    return filenames

def erase_results(window_obj, elements):
    """Helper function that sets each speicified window element to an empty string, effectively erasing them"""
    print("entered erase_results function")
    for elem in elements:
        print(elem)
        window_obj[elem].update("")
       

def delete_figure_agg(figure_agg):
    """Helper function to clear out previously drawn plots"""
    figure_agg.get_tk_widget().forget()
    plt.close('all')


#----------------------------------------------------------------------------
# GUI layout definitions
#----------------------------------------------------------------------------

# first of 2 column layouts
browse_file_column = [
    [
        sg.Text("Model File", size=(14, 1), font='bold'),
        sg.In(size=(50, 1), enable_events=True, key="-MODEL-"),
        sg.FileBrowse(),
    ],

    [
        sg.Text("Spectrum Folder", size=(14,1), font='bold'),
        sg.In(size=(50, 1), enable_events=True, key="-FOLDER-"),
        sg.FolderBrowse(),
    ],

    [sg.HSeparator(pad=(0, (20, 20)))],
   
    [sg.Text("File                   Modified Date", pad=((10, 0), 0))],

    [sg.Listbox(values=[], enable_events=True, size=(65, 20), key="-FILE LIST-")],
    
    [
        sg.Button("Sort File", pad=((10, 25), 0), key="-SORTFILE-"),
        sg.Button("Sort Date", key="-SORTDATE-"),
    ],

    [
        sg.Push(),
        sg.Button("View Detailed Results", pad=(10, 35), key="-MODE-"),
        sg.Push(),
    ],
]

# second of 2 columns.  will show name of spectral file chosen and an image of the spectrum
selected_file_column = [
    [sg.Text("Choose a spectral file from the list on the left:", font='Helvetica 14', key="-SPECTRUM SELECT-")],
    # initially displays user instructions, but afterward displays product type
    [sg.Text(size=(None, 1), font='Helvetica 18', justification='center', key="-TOUT-"), sg.Text(size=(None, 1), font='Helvetica 24 bold', justification='center', visible=False, key="-PRODUCT ID-")],
    # next two rows display PLS results
    [sg.Text(size=(None, 1), font='Helvetica 18', justification='center', key="-PROPERTY1-"), sg.Text(size=(None, 1), font='Helvetica 24 bold', justification='center', visible=False, key="-RESULT1-")],
    [sg.Text(size=(None, 1), font='Helvetica 18', justification='center', key="-PROPERTY2-"), sg.Text(size=(None, 1), font='Helvetica 24 bold', justification='center', visible=False, key="-RESULT2-")],
    # displays M-distance and spectral residual
    [
        sg.Text(size=(None, 1), font='Helvetica 18', justification='center', visible=True, key="-MDIST_LABEL-"), sg.Text(size=(None, 1), font='Helvetica 18 bold', justification='center', visible=True, pad=((0, 50), 0), key="-MDIST-"), 
        sg.Text(size=(None, 1), font='Helvetica 18', justification='center', visible=True, key="-RESID_LABEL-"), sg.Text(size=(None, 1), font='Helvetica 18 bold', justification='center', visible=True, key="-RESID-")
    ],

    [sg.Text(size=(None, 1), font='Helvetica 18', justification='center', visible=True, key="-PASS_FAIL_LABEL-"), sg.Text(size=(None, 1), font='Helvetica 24 bold underline', justification='center', visible=True, pad=(0, 25), key="-PASS_FAIL-")],

    # canvas for plotting spectra
    [sg.Canvas(key="-CANVAS-")],
]

# full layout
layout = [
    [
        sg.Column(browse_file_column),
        sg.VSeperator(),
        sg.Column(selected_file_column, element_justification='center'),
    ]
]

# main window
window = sg.Window("Rapid Product Release by Raman Spectroscopy", layout, resizable=True)

#----------------------------------------------------------------------------
# main event loop
#----------------------------------------------------------------------------
try:
    while True:
        event, values = window.read()

        if event == "-EXIT-" or event == sg.WIN_CLOSED:
            break

        elif event == "-MODEL-":   # user has selected a chemometric model file
            try:
                model_file = values["-MODEL-"]  # get file path for the model
                model = load_model(model_file)
            except:
                model = None

        elif event == "-FOLDER-":   # user has selected a spectrum folder
            folder = values["-FOLDER-"]
            filenames = pull_folder(folder, descending=descending)
            window["-FILE LIST-"].update(filenames)
            
        elif event == "-SORTFILE-":
            # do nothing if a folder hasn't been selected yet
            if folder:
                filenames = pull_folder(folder, sort_by_date=False, descending=descending)
                window["-FILE LIST-"].update(filenames)

        elif event == "-SORTDATE-":
            # do nothing if a folder hasn't been selected yet
            if folder:
                # toggle the descending flag so that it reverses the sort order each time button is pressed
                descending = not descending
                
                # update the displayed files
                filenames = pull_folder(folder, sort_by_date=True, descending=descending)
                window["-FILE LIST-"].update(filenames)

        elif event == "-MODE-":
            # toggle display mode between detailed view and pass/fail view and update the button text
            analyst_mode = not analyst_mode
            if analyst_mode:
                window["-MODE-"].update('View Detailed Results')
            else:
                window["-MODE-"].update('Summary View')
       
        elif event == "-FILE LIST-":  # a spectrum file was chosen from the listbox
            if figure_agg:  # a plot has already been drawn and needs to be cleared
                delete_figure_agg(figure_agg) 
            try:
                filename = os.path.join(values["-FOLDER-"], values["-FILE LIST-"][0][0])
                window["-SPECTRUM SELECT-"].update(visible=False)

                # convert spc file and create plot of spectrum
                figure, spec = convert_spc(filename)
                # add plot to the window
                figure_agg = draw_figure(window["-CANVAS-"].TKCanvas, figure)

                if model:
                    # identify product and analyze spectrum
                    product_type = model.identify_product(spec)
                    model_outputs = model.quantify(spec, product_type)

                    window["-TOUT-"].update("Product:")
                    window["-PRODUCT ID-"].update(product_type, visible=True)
                    
                    if model_outputs == "Model Not Trained":
                        # display model output (or lack thereof)
                        window["-PROPERTY1-"].update('Purity:')
                        window["-RESULT1-"].update(model_outputs, visible=True)

                        # erase result2
                        window["-PROPERTY2-"].update("")
                        window["-RESULT2-"].update("")
                        # erase M-distance information
                        window["-MDIST_LABEL-"].update("")
                        window["-MDIST-"].update("")
                        # erase spectral residual information
                        window["-RESID_LABEL-"].update("")
                        window["-RESID-"].update("")
                        # erase pass/fail information
                        window["-PASS_FAIL_LABEL-"].update("")
                        window["-PASS_FAIL-"].update("")

                    elif len(model_outputs[0]) == 1:  # this model only quantifies purity
                        # pull out result information
                        result = model_outputs[0][0]
                        prop = model_outputs[1]
                        m_dist = model_outputs[2]
                        residual = model_outputs[3]

                        # make pass/fail result label visible
                        window["-PASS_FAIL_LABEL-"].update('Release:', visible=True)
                        # determine pass/fail result
                        spec_limit = model.configs[product_type][prop]
                        m_dist_max = model.configs[product_type]['M-Distance Max']
                        residual_max = model.configs[product_type]['Spectral Residual Max']

                        if result >= spec_limit and m_dist <= m_dist_max and residual <= residual_max:
                            window["-PASS_FAIL-"].update('PASS')
                        else:
                            # right side of condition evaluates to None if product_type isn't in the config file,
                            # causing unconfigured products to default to "FAIL"
                            window["-PASS_FAIL-"].update('FAIL')

                        if analyst_mode:
                            # erase everything except pass/fail
                            window["-PROPERTY1-"].update("")
                            window["-RESULT1-"].update("")
                            window["-PROPERTY2-"].update("")
                            window["-RESULT2-"].update("")
                            window["-MDIST_LABEL-"].update("")
                            window["-MDIST-"].update("")
                            window["-RESID_LABEL-"].update("")
                            window["-RESID-"].update("")

                        else:
                            # show result and spectral metrics
                            window["-PROPERTY1-"].update(prop + ': ', visible=True)
                            window["-RESULT1-"].update('{0:.3f} w%'.format(result), visible=True)
                            window["-MDIST_LABEL-"].update("M-Distance: ", visible=True)
                            window["-MDIST-"].update('{:.2f}'.format(m_dist), visible=True)
                            window["-RESID_LABEL-"].update("Spectral Residual: ", visible=True)
                            window["-RESID-"].update('{:.2e}'.format(residual), visible=True)
                        
                            # erase unused result2 
                            window["-PROPERTY2-"].update("")
                            window["-RESULT2-"].update("")

                    elif len(model_outputs[0]) == 2:  # this model quantifies both lights and heavies
                        result1, result2 = model_outputs[0]
                        prop1, prop2 = model_outputs[1]
                        m_dist = model_outputs[2]
                        residual = model_outputs[3]

                        # make pass/fail result label visible
                        window["-PASS_FAIL_LABEL-"].update('Release:', visible=True)

                        # determine pass/fail result
                        spec_limit1 = model.configs[product_type][prop1]
                        spec_limit2 = model.configs[product_type][prop2]
                        m_dist_max = model.configs[product_type]['M-Distance Max']
                        residual_max = model.configs[product_type]['Spectral Residual Max']

                        if result1 <= spec_limit1 and result2 <= spec_limit2 and m_dist <= m_dist_max and residual <= residual_max: 
                            window["-PASS_FAIL-"].update('PASS')
                        else:
                            # right side of condition evaluates to None if product_type isn't in the config file,
                            # causing unconfigured products to default to "FAIL"
                            window["-PASS_FAIL-"].update('FAIL')

                        if analyst_mode:
                            # erase everything except pass/fail
                            window["-PROPERTY1-"].update("")
                            window["-RESULT1-"].update("")
                            window["-PROPERTY2-"].update("")
                            window["-RESULT2-"].update("")
                            window["-MDIST_LABEL-"].update("")
                            window["-MDIST-"].update("")
                            window["-RESID_LABEL-"].update("")
                            window["-RESID-"].update("")

                        else:
                            # show first result
                            window["-PROPERTY1-"].update(prop1 + ': ', visible=True)
                            window["-RESULT1-"].update('{0:.3f} w%'.format(result1), visible=True)
                            # show second result
                            window["-PROPERTY2-"].update(prop2 + ': ', visible=True)
                            window["-RESULT2-"].update('{0:.3f} w%'.format(result2), visible=True)
                        
                            # show spectral metrics
                            window["-MDIST_LABEL-"].update("M-Distance: ", visible=True)
                            window["-MDIST-"].update('{:.2f}'.format(m_dist), visible=True)
                            window["-RESID_LABEL-"].update("Spectral Residual: ", visible=True)
                            window["-RESID-"].update('{:.2e}'.format(residual), visible=True)

                else:
                    window["-TOUT-"].update('Select a model file to analyze this spectrum.')

            except:
                pass

except Exception as e:
    tb = traceback.format_exc()
    sg.Print("An error has occurred.  Here is the info:", e, tb)
    sg.popup_error("AN EXCEPTION OCCURRED!", e, tb)

window.close()