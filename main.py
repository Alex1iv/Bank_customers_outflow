
import sys
from streamlit.web import cli as stcli


if __name__ == "__main__":
    
    # get f1-score:
    #--------------------
    #get_data() 
    
    
    # otherwise get a prediction in a web page 
    #--------------------   
    print('The program returns prediction by given data')
    #start_streamlit()
    
    sys.argv = ["streamlit", "run", "application.py"]
    
    sys.exit(stcli.main())
    
   