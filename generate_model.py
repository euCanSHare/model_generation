import sys,os,logging,random,pickle,datetime,joblib
import pandas as pd
import numpy as np

#Preprocessing
from sklearn.preprocessing._data import MinMaxScaler,RobustScaler

#Feature selection and dimensionality reduction
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.decomposition import PCA, NMF
#from mlxtend.feature_selection import SequentialFeatureSelector as SFS

#Machine Learning Models
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import tree



#Model selection and Validation 
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline


#Performance Metrics
from sklearn.metrics import classification_report, accuracy_score, make_scorer
from sklearn.metrics import auc
from sklearn.metrics import plot_roc_curve
from sklearn.inspection import permutation_importance


#plot library
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
plt.style.use("seaborn")


#dummy dataset
from sklearn.datasets import load_breast_cancer

#pdf template
import pdfkit
from pdf.summary_template import *


# set logging config

logging.basicConfig(
    filename='machine_learning_report_log.log',
    level=logging.INFO,
    format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)

#VRE logger
from utils import logger



def save_pdf(path):
    
    options = {   
        'disable-smart-shrinking': '',
        'quiet': '',
        'margin-top': '0.1in',
        'margin-right': '0in',
        'margin-bottom': '0in',
        'margin-left': '0in',
    
    }
    
    pdfkit.from_string(body,path, options = options,css = os.path.join(os.getcwd(),'pdf','style.css'))
    
    
def generate_random_color():
    
    c = tuple(np.random.randint(256, size=4)/255)
    c_eush = (c[0],c[1]/5,1-c[0],np.min([c[3]*3, 1.0]))
    return c_eush

def run(file_dataset = None, 
        classifier = 'logistic_regression', 
        max_features = 10, 
        n_folds = 5, 
        output_file = 'default_summary.pdf'):
    
    logging.info('Running generate_model.py')
    logging.info('Current working directory: {}'.format(os.getcwd()))
    logging.info('classifier {}'.format(classifier))
    logging.info('max_features {}'.format(max_features))
    logging.info('n_folds {}'.format(n_folds))
    logger.info('Running generate_model.py')
    logger.info('Current working directory: {}'.format(os.getcwd()))
    logger.info('classifier {}'.format(classifier))
    logger.info('max_features {}'.format(max_features))
    logger.info('n_folds {}'.format(n_folds))

    
    
    classifiers = {
        'logistic_regression': LogisticRegression(max_iter=2000,random_state = 42)
    }

    

    execution_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    results = pd.DataFrame(columns=('file_in', 
                                    'n_features',
                                    'mean_acc', 
                                    'std_acc',
                                    'mean_f1', 
                                    'mean_pre', 
                                    'mean_rec', 
                                    'clf'))

    if file_dataset is not None:
        
        logging.info('file_dataset loaded correctly{}'.format(file_dataset))

        df = pd.read_csv(file_dataset)
        data = np.array(df) 
        
        ids = data[:,0] 
        y   = data[:,1]        
        X   = data[:,2:]
        features_names = list(df.columns)[2:]
        
    else:

        logging.info('file_dataset {} not found, loading dummy dataset'.format(file_dataset))
        logger.info('file_dataset {} not found, loading dummy dataset'.format(file_dataset))

        X, y = load_breast_cancer(return_X_y=True)
        X = np.array(X)
        y = np.array(y)
        
        features_names = list(load_breast_cancer(return_X_y=False)['feature_names'])

    """Battery of classifiers with model-agnostic feature selection"""
    for n_features in list(range(1,max_features)):      
        
        logging.info('Performing analysis with the {} best features'.format(n_features))
        logger.info('Performing analysis with the {} best features'.format(n_features))

   

        # Variables for average classification report
        originalclass = []
        predictedclass = []
        
        
        #Make our customer score
        def classification_report_with_accuracy_score(y_true, y_pred):
            originalclass.extend(y_true)
            predictedclass.extend(y_pred)
            return accuracy_score(y_true, y_pred) # return accuracy score
        
        cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        
        if n_features > X.shape[1]:
            continue
      
        pipe = Pipeline([
              ('scaler', MinMaxScaler()),
              ('reduce_dim', SelectKBest(chi2, k = n_features)),
              ('classification', classifiers[classifier])
            ])

        # Nested CV with parameter optimization                
        folds_score = cross_val_score(pipe, X=X, y=y, cv=cv, scoring=make_scorer(classification_report_with_accuracy_score))

        # Average values in classification report for all folds in a K-fold Cross-validation  
        cl_report = classification_report(originalclass, predictedclass, output_dict = True)
        
        #joblib.dump(sfs1, './results/'+execution_time+'/sfs'+str(sfs_k)+'_'+clf[1]+file_dataset+name_feids+'_sfsobj.model')
        #joblib.dump(clf[0],  './results/'+execution_time+'/sfs'+str(sfs_k)+'_'+clf[1]+file_dataset+name_feids+'_clfobj.model')
        results = results.append(pd.Series({'file_in':file_dataset,
                                            'n_features':str(n_features),
                                            'std_acc':np.std(folds_score),
                                            'mean_acc':cl_report['accuracy'], 
                                            'mean_f1':cl_report['macro avg']['f1-score'], 
                                            'mean_pre':cl_report['macro avg']['precision'], 
                                            'mean_rec':cl_report['macro avg']['recall'], 
                                            'clf':classifier}),
                                            ignore_index = True
            )
    
    
    
    
    
    #TO-DO select the minimum number of features when the same accuracy is reached 
    best_n_features = int(results[results.mean_acc == results.mean_acc.max()].iloc[0].n_features)
    logging.info('Optimal number of features found: {}'.format(best_n_features))
    logger.info('Optimal number of features found: {}'.format(best_n_features))


    
    
    pipe = Pipeline([
      ('scaler', MinMaxScaler()),
      ('reduce_dim', SelectKBest(chi2, k = best_n_features)),
      ('classification', classifiers[classifier])
    ])
            

    logging.info('Generating ROC auc plot and computing feature importances...')
    logger.info('Generating ROC auc plot and computing feature importances...')

    
    tprs = []
    aucs = []
    importances_mean = []
    importances = []
    mean_fpr = np.linspace(0, 1, 100)



    fig, ax = plt.subplots()

    for i, (train, test) in enumerate(cv.split(X, y)):
        pipe.fit(X[train], y[train])
        viz = plot_roc_curve(pipe, X[test], y[test],
                             name='ROC fold {}'.format(i),
                             alpha=0.3, lw=1, ax=ax)
        interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        aucs.append(viz.roc_auc)
        
        perm_imp = permutation_importance(pipe, X[test], y[test], n_repeats=10,
                                random_state=42, n_jobs=-1)
        
        importances_mean.append(perm_imp.importances_mean)
        importances.append(perm_imp.importances) # importances per feature per subject
    
    ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
            label='Chance', alpha=.8)
    
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    ax.plot(mean_fpr, mean_tpr, color='b',
            label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
            lw=2, alpha=.8)
    
    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                    label=r'$\pm$ 1 std. dev.')
    
    ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05])
                                     
           #,title="ROC for CV{} with {} optimal selected features".format(n_folds,best_n_features))
    ax.legend(loc="lower right")
    fig.set_size_inches(6, 6)

    fig.tight_layout()

    #plt.show()
    path_roc = os.path.join(os.getcwd(),'pdf','figures',"roc-curve.png")
    logging.info('Saving ROC AUC plot in {}...'.format(path_roc))
    logger.info('Saving ROC AUC plot in {}...'.format(path_roc))

    fig.savefig(path_roc)
    
    
    
    
    logging.info('Generating feature importances plot...')
    logger.info('Generating feature importances plot...')

    mean_relevances = np.mean(np.array(importances_mean),axis = 0)
    sorted_idx = np.squeeze(np.flip(mean_relevances.argsort()))[:best_n_features]
    importances = np.mean(importances,axis = 2).T
    
    names = [features_names[ind] for ind in sorted_idx]

    fig, ax = plt.subplots()
    bplot = ax.boxplot(importances[sorted_idx].T,
               vert=True, labels=list(range(len(names))),patch_artist=True)
    #ax.set_title("Selected features importance and variance across folds")
    ax.set_ylabel('Importance and variance across folds')
    ax.set_xlabel('Feature Number')
    
    
    legend_handles = []
    for feat in range(len(names)):
        name = names[feat]
        color = generate_random_color()
        bplot['boxes'][feat].set_facecolor(color)
        legend_handles.append(mpatches.Patch( label=name,color=color))
        
        
    ax.legend(loc="upper right",handles=legend_handles)
    fig.set_size_inches(6, 6)
    fig.tight_layout()

    #plt.show()

    path_rel = os.path.join(os.getcwd(),'pdf','figures',"feat-rel.png")
    logging.info('Saving feature importances plot in {}...'.format(path_rel))
    logger.info('Saving feature importances plot in {}...'.format(path_rel))

    fig.savefig(path_rel)
    
    
    try:
        logging.info('Generating pdf summary...')   
        logger.info('Generating pdf summary...')        
     
        #TO-DO passing a dictionary to replace key information in the html string
        save_pdf(output_file)
 
    except Exception:
        logging.info(sys.exc_info()[1])
        logger.info(sys.exc_info()[1])
    
    logging.info('Pdf summary generated in {}...'.format(output_file))  
    logger.info('Pdf summary generated in {}...'.format(output_file))   
 

run()


         
       


