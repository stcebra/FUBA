from mpi4py import MPI

comm = MPI.COMM_WORLD
SIZE = comm.Get_size() - 1
free_nodes = [True for _ in range(SIZE)]
if free_nodes:
    free_nodes[0] = False

def get_free(local_model_dicts, backdoor_model_dicts):
    global free_nodes
    while 1:
        try:
            i = free_nodes.index(True)
            break
        except ValueError:
            try_recv_models(local_model_dicts)
            try_recv_backdoor_models(backdoor_model_dicts)
            try_free_nodes()
    free_nodes[i] = False
    return i

    
def recover_free(i):
    global free_nodes
    free_nodes[i] = True


def wait_free():
    comm.Probe(tag=0)
    data = comm.recv(tag=0)
    return data


def try_free_nodes():
    while comm.Iprobe(tag=0):
        node_idx = comm.recv(tag=0)
        recover_free(node_idx)


def try_recv_models(model_dicts):
    while comm.Iprobe(tag=1):
        idx, model_dict = comm.recv(tag=1)
        model_dicts[idx] = model_dict


def try_recv_backdoor_models(model_dicts):
    while comm.Iprobe(tag=2):
        idx, model_dict = comm.recv(tag=2)
        model_dicts[idx] = model_dict


def allocate_task(node, tag):
    for i in range(1,SIZE):
        comm.isend(node == i, i, tag)


def allocate_double_task(node1, node2, tag):
    for i in range(1,SIZE):
        if i != node1 and i != node2:
            data = False
        elif i == node1:
            data = (0,node2)
        else:
            data = (1,node1)
        comm.isend(data, i, tag)


def recv_task(tag):
    return comm.recv(None, 0, tag)


