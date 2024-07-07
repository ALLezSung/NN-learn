def show_model(model):
    ''' 
        显示模型结构以及参数信息
    Args:
        model: nn.Module, 待显示的模型
    Returns:
        None
    '''
    print(model)
    print('Model parameters:')
    for name, param in model.named_parameters():
        print(f'Layer: {name} | Size: {param.size()} | Values : {param[:2]}')
    print('\n')

    return None