"""Compile the existing ordered CUDA kernels into one native step submission.

Taichi 1.7's public graph dispatcher cannot bind class template arguments.
This small adapter binds each compiled variant first, then adds its native
kernel to the graph. Solver fields and kernel barriers remain unchanged.
"""
import taichi as ti


class CudaStepGraph:
    def __init__(self, solver):
        self.solver = solver
        builder = ti.graph.GraphBuilder()

        def dispatch(method, *templates):
            kernel = method._primal
            args = (method._kernel_owner, *templates)
            key = kernel.ensure_compiled(*args)
            builder._graph_builder.dispatch(kernel.compiled_kernels[key], [])

        s, c = solver, solver.config
        if c.buoyancy_assist and c.coupling == "two_way":
            dispatch(s.buoyancy.begin)
        dispatch(s.materials.predict)
        if s.bodies.count:
            dispatch(s.bodies.predict)
        if s.boundary_density is not None:
            for kernel in s.boundary_density.build_kernels:
                dispatch(kernel)
        for iteration in range(c.iterations):
            if c.transfer_mode == "incremental" and iteration > 0:
                dispatch(s.prepare_increment)
                dispatch(s.transfers.p2g_increment)
                if s.container is not None:
                    dispatch(s.grid_update_increment)
                else:
                    dispatch(s.transfers.incremental_boundary)
            else:
                dispatch(s.prepare_iteration)
                dispatch(s.transfers.p2g)
                if s.container is not None and not s.has_elastic:
                    dispatch(s.grid_update)
                else:
                    dispatch(s.transfers.boundary)
            dispatch(s.transfers.g2p_with_feedback, s.bodies)
            if s.container is not None:
                dispatch(s.project_interfaces)
            else:
                if s.bodies.count and c.coupling == "two_way":
                    dispatch(s.bodies.apply_feedback)
                if s.bodies.count and c.rigid_contact:
                    dispatch(s.contacts.project)
                    if c.coupling == "two_way":
                        dispatch(s.bodies.apply_contact_feedback)
                    if s.contacts.has_continuum:
                        if s.contacts.can_reuse_mass:
                            dispatch(s.contacts.scatter_contact_increment)
                            dispatch(s.contacts.gather_contact_with_mass, s.transfers.mass)
                        else:
                            dispatch(s.contacts.scatter_contact_correction)
                            dispatch(s.contacts.gather_contact_correction)
        if c.buoyancy_assist and c.coupling == "two_way":
            dispatch(s.buoyancy.apply)
        dispatch(s.materials.integrate)
        if s.bodies.count:
            dispatch(s.bodies.advance_pose)
            dispatch(s.bodies.reconstruct)
            if s.container is not None:
                for _ in range(2):
                    if s.box_pairs is not None:
                        dispatch(s.box_pairs.project)
                    dispatch(s.container.resolve_bodies)
                dispatch(s.bodies.reconstruct)
        dispatch(s.check_state)
        self.graph = builder.compile()

    def run(self):
        # Prescribed mouse targets are integrated once per physical dt on the
        # host, exactly as in the ordinary path, before native step submission.
        if self.solver.container is not None:
            self.solver.container.begin_step()
        self.graph.run({})
