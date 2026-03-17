import streamlit as st


def init_state():
    defaults = dict(
        page="Imputation",
        transformer_choice="LaBSE",
        mode="Incremental Imputation",
        raw_df=None,
        working_df=None,
        imputed_df=None,
        imputed_mask=None,
        selected_cols=[],
        strategy="SENTI",
        other_strategy="mean",
        append_highlight=None,
        flow_state="idle",      # "idle" | "post_impute_prompt" | "append_phase" | "finished"
        _doc=False,
        iter_k=1,
        last_imputed_iter=0,
        source_snapshot=None,
        pre_append_snapshot=None,
        append_history=[],
    )
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v.copy() if isinstance(v, (list, dict)) else v


class _SSProxy:
    def __getattr__(self, name):
        return st.session_state.get(name, None)

    def __setattr__(self, name, value):
        st.session_state[name] = value

    def get(self, key, default=None):
        return st.session_state.get(key, default)


ss = _SSProxy()
