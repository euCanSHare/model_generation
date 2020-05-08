import os 
import pathlib

# TO-DO add div id's and styles to the css file.
# TO-DO cleaning and indentation
# TO-DO margin issues
# TO-DO replace key fields in html tables with real information (see generated_model.py>save_pdf()) 

body = \
"""
<html>
    <head>
        <style>
        
            body{
                width: 20.6cm;
                height: 29.5cm;
                font-family:    Arial, Helvetica, sans-serif;
            }
            
    
        </style>
    
    </head>
    
    <body>
    
        <div style="border: 0px solid #1C6EA4;width: 100%; height: 100%;position: relative;box-sizing: border-box">
        
            <div style="border: 1px solid #1C6EA4;width: 100%; height: 12%;position: relative">
            
                <div style="width: 18%; height: 100%;display:inline-block;text-align:center">              
                      <img src='eush_logo.png' style="height: 95%;padding:2px">              
                </div>
                        
              
                <div style="width: 81.6%; height: 100%;display:inline-block; float:right">       
                    <table class="blueTable">
                    
                        <thead>
                            <tr>
                                <th>CARDIAC AI REPORT</th>
                                <th></th>
                            </tr>
                        </thead>
                        
                        <tbody>
                            <tr>
                                <td>USER</td>
                                <td>Test User</td>
                            </tr>
                            <tr>
                                <td>DATE</td>
                                <td>Tuesday, April 21, 2020 15:33 PM</td>
                            </tr>
                            
                            <tr>
                                <td>DATASET</td>
                                <td>ACDC Dataset</td>
                            </tr>
                            <tr>
                                <td>PATHOLOGY</td>
                                <td>Myocardial Infarction</td>
          
                            </tr>
                        </tbody>
                        </tr>
                    </table>
                </div>
            </div>
            
            
            <div style="border: 1px solid #1C6EA4;width: 100%; height: 31%;position: relative">
                
                <div style="width: 49%; height: 100%;float:left;text-align:center;;padding:5px;box-sizing: border-box">
    
                    <img src='mesh_heart.png' style="height: 80%;;padding:3px">
                    <div style="text-align:center;font-family: "Roboto", sans-serif;">
                        <p style="text-align:center;font-family: "Roboto", sans-serif;">
                            Automatic deep learning 3D contours generated for the first subject of the dataset based on CMR.
                        </p>
                    </div>
                  
                </div>
            
                <div style="width: 50%; height: 100%;float:right;;padding:5px;box-sizing: border-box">
                    <table class="blueTable" style='font-size: 10px'>
                    
                        <thead>
                        <tr>
                            <th>PIPELINE SUMMARY</th>
                            <th></th>
                        </tr>
                        </thead>
                        
                        <tbody>
                        <tr>
                            <td>RADIOMIC VERSION</td>
                            <td>Pyradiomics 0.2.0</td>
                        </tr>
                        
                        <tr>
                            <td>NUMBER OF CONTROLS</td>
                            <td>20</td>
                        </tr>
                        <tr>
                            <td>NUMBER OF CASES</td>
                            <td>20</td>
                        </tr>
                        <tr>
                            <td>REGIONS OF INTEREST</td>
                            <td>Left Ventricle, Right Ventricle, Myocardium</td>
      
                        </tr>
                        <tr>
                            <td>NUMBER OF FEATURES</td>
                            <td>315</td>
                        </tr>
                        <tr>
                            <td>MACHINE LEARNING TECHNIQUE</td>
                            <td>Logistic Regression</td>
                        </tr>
                        <tr>
                            <td>FEATURE SELECTION</td>
                            <td>Chi Square</td>
                        </tr>
                        <tr>
                            <td>RELEVANT FEATURES</td>
                            <td>5 / 315</td>
                        </tr>
                        </tbody>
                    </table>

              </div>
            
            </div>
            
            <div style="border: 1px solid #1C6EA4;width: 100%; height: 35%;position: relative;text-align:center">
                    <table class="blueTable">
                
                        <thead>
                            <tr>
                                <th style="text-align:center">SELECTED FEATURES IMPORTANCE AND VARIANCE ACROSS FOLDS</th>
                                <th style="text-align:center">CROSS-VALIDATED ROC WITH OPTIMAL SELECTED FEATURES</th>
                            </tr>
                        </thead>
      
                    </table>
            
            
                    <div style="width: 49%; height: 100%;display:inline-block;text-align:center">
              
                          <img src='feat-rel.png' style="max-height: 95%;width: 95%;padding:5px">
                  
                    </div>
                    
                    <div style="width: 49%; height: 100%;display:inline-block;text-align:center">
              
                          <img src='roc-curve.png' style="max-height: 95%;width: 95%;padding:5px">
                
                    </div>
            
            </div>
            
            
            <div style="border: 1px solid #1C6EA4;width: 100%; height: 20%;position: relative;">
            
                <img src='eush_footer.png' style="height: 100%;width: 100%;">
            
            </div>


        </div>
    </body>
</html>
"""


#Replacing images with location in disk
body = body.replace('eush_logo.png',pathlib.Path(os.path.join(os.getcwd(),'pdf','figures','eush_logo.png')).as_uri())
body = body.replace('eush_footer.png',pathlib.Path(os.path.join(os.getcwd(),'pdf','figures','eush_footer.png')).as_uri())
body = body.replace('feat-rel.png',pathlib.Path(os.path.join(os.getcwd(),'pdf','figures','feat-rel.png')).as_uri())
body = body.replace('roc-curve.png',pathlib.Path(os.path.join(os.getcwd(),'pdf','figures','roc-curve.png')).as_uri())
body = body.replace('mesh_heart.png',pathlib.Path(os.path.join(os.getcwd(),'pdf','figures','mesh_heart.png')).as_uri())
