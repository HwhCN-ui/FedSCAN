#!/usr/bin/env python3
# Copyright (c) JLU, NODI, Huwenhan. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");

defense = {'enabled': False}
stage1 = {'lambda_m': 0.1, 'max_iter': 10, 'tol': 1e-06, 'loss_type': 'exponential'}
stage2 = {'c': 1.0, 'svd_mode': 'thin', 'benign_init_round': 0}
