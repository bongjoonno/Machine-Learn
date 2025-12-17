from src.machine_learn.imports import fetch_ucirepo 
  
wine_quality = fetch_ucirepo(id=186) 

wine_x = wine_quality.data.features 
wine_y = wine_quality.data.targets 
  
wine_columns_to_scale = list(wine_x.columns)