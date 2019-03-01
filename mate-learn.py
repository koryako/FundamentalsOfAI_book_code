def get_params(module, memo=None, pointers=None):
    """ Returns an iterator over PyTorch module parameters that allows to update parameters
        (and not only the data).
    ! Side effect: update shared parameters to point to the first yield instance
        (i.e. you can update shared parameters and keep them shared)
    Yields:
        (Module, string, Parameter): Tuple containing the parameter's module, name and pointer
    """
    if memo is None:
        memo = set()
        pointers = {}
    for name, p in module._parameters.items():
        if p not in memo:
            memo.add(p)
            pointers[p] = (module, name)
            yield module, name, p
        elif p is not None:
            prev_module, prev_name = pointers[p]
            module._parameters[name] = prev_module._parameters[prev_name] # update shared parameter pointer
    for child_module in module.children():
        for m, n, p in get_params(child_module, memo, pointers):
            yield m, n, p


class MetaLearner(nn.Module):
    """ Bare Meta-learner class
        Should be added: intialization, hidden states, more control over everything
    """
    def __init__(self, model):
        super(MetaLearner, self).__init__()
        self.weights = Parameter(torch.Tensor(1, 2))

    def forward(self, forward_model, backward_model):
        """ Forward optimizer with a simple linear neural net
        Inputs:
            forward_model: PyTorch module with parameters gradient populated
            backward_model: PyTorch module identical to forward_model (but without gradients)
              updated at the Parameter level to keep track of the computation graph for meta-backward pass
        """
        f_model_iter = get_params(forward_model)
        b_model_iter = get_params(backward_model)
        for f_param_tuple, b_param_tuple in zip(f_model_iter, b_model_iter): # loop over parameters
            # Prepare the inputs, we detach the inputs to avoid computing 2nd derivatives (re-pack in new Variable)
            (module_f, name_f, param_f) = f_param_tuple
            (module_b, name_b, param_b) = b_param_tuple
            inputs = Variable(torch.stack([param_f.grad.data, param_f.data], dim=-1))
            # Optimization step: compute new model parameters, here we apply a simple linear function
            dW = F.linear(inputs, self.weights).squeeze()
            param_b = param_b + dW
            # Update backward_model (meta-gradients can flow) and forward_model (no need for meta-gradients).
            module_b._parameters[name_b] = param_b
            param_f.data = param_b.data


def train(forward_model, backward_model, optimizer, meta_optimizer, train_data, meta_epochs):
  """ Train a meta-learner
  Inputs:
    forward_model, backward_model: Two identical PyTorch modules (can have shared Tensors)
    optimizer: a neural net to be used as optimizer (an instance of the MetaLearner class)
    meta_optimizer: an optimizer for the optimizer neural net, e.g. ADAM
    train_data: an iterator over an epoch of training data
    meta_epochs: meta-training steps
  To be added: intialization, early stopping, checkpointing, more control over everything
  """
  for meta_epoch in range(meta_epochs): # Meta-training loop (train the optimizer)
    optimizer.zero_grad()
    losses = []
    for inputs, labels in train_data:   # Meta-forward pass (train the model)
      forward_model.zero_grad()         # Forward pass
      inputs = Variable(inputs)
      labels = Variable(labels)
      output = forward_model(inputs)
      loss = loss_func(output, labels)  # Compute loss
      losses.append(loss)
      loss.backward()                   # Backward pass to add gradients to the forward_model
      optimizer(forward_model,          # Optimizer step (update the models)
                backward_model)
    meta_loss = sum(losses)             # Compute a simple meta-loss
    meta_loss.backward()                # Meta-backward pass
    meta_optimizer.step()        