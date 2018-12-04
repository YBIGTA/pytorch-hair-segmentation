from networks import unet

def choose_network(network_name, args):
    model = None
    if network_name == 'unet':
        model = unet(args.ic, args.oc, args.class)
    return model