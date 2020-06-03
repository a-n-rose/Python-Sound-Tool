



class BaseModel:
    '''Base class to hold basic information necessary for all models. Should model 
    related classes have parent/child relations with feature extraction classes? 
    Perhaps better if apply an analysis on the feature data and save with model info.
    
    pathway
    training data
    duration
    model architecture
    model type?
    data type?
    '''
    pass

class SpecificModel(BaseModel):
    '''Necessary? Perhaps for establishing defaults for specific types of models
    
    metrics
    loss 
    optimizer
    augmentation
    learning rates
    '''
    pass
