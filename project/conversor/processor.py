import torch
import torch.nn.functional as F
from project.core.application import Application
from project.dto.tts_dto import KokoroTtsDto, RvcDTO, RvcTtsDTO
import logging
import numpy as np


class VoiceConverterProcessor:
    def __init__(self, model):
        self.model = model
        self.app = Application()

    async def convert_voice(self, src_wav, target_embedding):
        print(f"[VoiceConverterProcessor] Iniciando conversão de voz. Shape do áudio fonte: {src_wav.shape}")
        print(f"[VoiceConverterProcessor] Shape do embedding alvo: {target_embedding.shape}")
        result = self.voice_conversion_with_target_se(src_wav, target_embedding)
        print(f"[VoiceConverterProcessor] Conversão concluída. Shape do resultado: {result.shape if result is not None else 'None'}")
        if result is None:
            print("[VoiceConverterProcessor] A conversão retornou None")
        return result
        
    @torch.inference_mode()
    def voice_conversion_with_target_se(self, src, tgt_se):
        print("[VoiceConverterProcessor] Extraindo source embedding e spectrograma")
        src_se, src_spec = self.model.extract_se(src)
        print(f"[VoiceConverterProcessor] Source embedding shape: {src_se.shape}, Source spec shape: {src_spec.shape}")
        # compute and show similarity between source and target embeddings
        try:
            # ensure embeddings have shape [1, C, 1] or [1, C]
            src_vec = src_se.squeeze(-1)
            tgt_vec = tgt_se.squeeze(-1)
            # normalize before similarity
            src_norm = F.normalize(src_vec, p=2, dim=1)
            tgt_norm = F.normalize(tgt_vec, p=2, dim=1)
            cosine = F.cosine_similarity(src_norm, tgt_norm, dim=1).item()
            l2 = torch.norm(src_vec - tgt_vec).item()
            print(f"[VoiceConverterProcessor] Embedding similarity -> cosine: {cosine:.4f}, L2: {l2:.4f}")
            # additional diagnostics: model config and tensor devices/values
            try:
                # unwrap nested wrappers to find underlying model
                model_obj = self.model
                depth = 0
                while hasattr(model_obj, 'model') and depth < 5:
                    model_obj = getattr(model_obj, 'model')
                    depth += 1
                zero_g = getattr(model_obj, 'zero_g', None)
                print(f"[VoiceConverterProcessor] Model zero_g: {zero_g}")
            except Exception:
                print("[VoiceConverterProcessor] Could not read model zero_g flag")

            try:
                print(f"[VoiceConverterProcessor] src_se device: {src_se.device}, dtype: {src_se.dtype}")
                print(f"[VoiceConverterProcessor] tgt_se device: {tgt_se.device}, dtype: {tgt_se.dtype}")
                print(f"[VoiceConverterProcessor] src_spec device: {src_spec.device}, dtype: {src_spec.dtype}")
                print(f"[VoiceConverterProcessor] src_se stats mean/std/min/max: {src_se.mean().item():.4f}/{src_se.std().item():.4f}/{src_se.min().item():.4f}/{src_se.max().item():.4f}")
                print(f"[VoiceConverterProcessor] tgt_se stats mean/std/min/max: {tgt_se.mean().item():.4f}/{tgt_se.std().item():.4f}/{tgt_se.min().item():.4f}/{tgt_se.max().item():.4f}")
                print(f"[VoiceConverterProcessor] src and tgt equal: {torch.allclose(src_se, tgt_se, atol=1e-6)}")
            except Exception as e:
                print(f"[VoiceConverterProcessor] Could not compute extended embedding diagnostics: {e}")

            # Optional quick test: replace target embedding with a random normalized vector to check effect
            try:
                import os
                if os.environ.get('RVC_TEST_RANDOM_GT', '0') == '1':
                    device = getattr(self.model, 'device', None)
                    # determine device from src_spec if not on model
                    if device is None:
                        device = src_spec.device
                    C = src_vec.shape[1]
                    rand = torch.randn((1, C), device=device)
                    rand = F.normalize(rand, p=2, dim=1).unsqueeze(-1)
                    tgt_se = rand
                    tgt_vec = tgt_se.squeeze(-1)
                    print('[VoiceConverterProcessor] Replaced target SE with random vector for quick test')
            except Exception as e:
                print(f"[VoiceConverterProcessor] Could not apply random gt override: {e}")
        except Exception as e:
            print(f"[VoiceConverterProcessor] Could not compute embedding similarity: {e}")

        # keep original SE shapes returned by extract_se (e.g., [1, C, 1])
        try:
            target_device = getattr(src_spec, 'device', None)
            target_dtype = getattr(src_spec, 'dtype', None)
            if isinstance(src_se, torch.Tensor):
                if target_dtype is not None:
                    src_se = src_se.to(target_dtype)
                if target_device is not None:
                    src_se = src_se.to(target_device)
            if isinstance(tgt_se, torch.Tensor):
                if target_dtype is not None:
                    tgt_se = tgt_se.to(target_dtype)
                if target_device is not None:
                    tgt_se = tgt_se.to(target_device)
        except Exception:
            pass

        aux_input = {"g_src": src_se, "g_tgt": tgt_se}
        print("[VoiceConverterProcessor] Iniciando inferência do modelo")

        # use a diagnostic wrapper around model.inference to log auxiliary info and compare outputs
        audio = self._run_inference_with_diagnostics(
            self.model,
            src_spec,
            aux_input,
            src_wave_numpy=src if isinstance(src, (np.ndarray,)) else None,
        )

        # Optional: run a second inference with a random target SE and compare outputs
        try:
            import os
            if os.environ.get('RVC_DIAG_COMPARE_RANDOM', '0') == '1':
                # build random normalized SE matching dims
                src_vec = src_se.squeeze(-1)
                C = src_vec.shape[1]
                device = src_spec.device
                rand = torch.randn((1, C), device=device)
                rand = F.normalize(rand, p=2, dim=1).unsqueeze(-1)
                aux_rand = {"g_src": src_se, "g_tgt": rand}
                audio_rand = self._run_inference_with_diagnostics(self.model, src_spec, aux_rand)
                # try to extract wavs for comparison
                out_a = None
                out_b = None
                try:
                    out_a = audio['model_outputs'][0, 0].data.cpu().float().numpy()
                    out_b = audio_rand['model_outputs'][0, 0].data.cpu().float().numpy()
                    n = min(out_a.shape[0], out_b.shape[0])
                    a = out_a[:n].astype(np.float32)
                    b = out_b[:n].astype(np.float32)
                    mse_ab = float(np.mean((a - b) ** 2))
                    corr_ab = float(np.corrcoef(a.flatten(), b.flatten())[0, 1]) if n > 1 else float('nan')
                    print(f"[VoiceConverterProcessor] Diagnostic compare: MSE between original-target and random-target outputs: {mse_ab:.6e}, Corr: {corr_ab:.6f}")
                except Exception:
                    print("[VoiceConverterProcessor] Diagnostic compare: could not compare outputs (missing model_outputs)")
        except Exception:
            pass
        print(f"[VoiceConverterProcessor] Inferência concluída. Keys do resultado: {audio.keys()}")

        if "model_outputs" not in audio:
            print("[VoiceConverterProcessor] Nenhum output do modelo encontrado")
            return None

        result = audio["model_outputs"][0, 0].data.cpu().float().numpy()
        print(f"[VoiceConverterProcessor] Áudio convertido com sucesso. Shape do resultado: {result.shape}")
        return result

    def _run_inference_with_diagnostics(self, model, src_spec, aux_input, src_wave_numpy=None):
        logger = logging.getLogger("logger")
        try:
            # Log aux_input keys
            try:
                logger.debug(f"[Diagnostic] aux_input keys: {list(aux_input.keys())}")
            except Exception:
                logger.debug("[Diagnostic] aux_input not a dict or keys not accessible")

            # Get model parameter dtype/device
            # unwrap nested wrappers until we find a torch.nn.Module or exhaust
            model_obj = model
            try:
                depth = 0
                while hasattr(model_obj, 'model') and depth < 5:
                    model_obj = getattr(model_obj, 'model')
                    depth += 1
            except Exception:
                pass
            logger.debug("[Diagnostic] underlying model object type: %s", str(type(model_obj)))

            param_dtype = None
            param_device = None
            try:
                if hasattr(model_obj, 'parameters'):
                    params_iter = model_obj.parameters()
                    first_param = next(params_iter, None)
                    if first_param is not None:
                        param_dtype = first_param.dtype
                        param_device = first_param.device
                        # count params (best-effort)
                        try:
                            cnt = 1 + sum(1 for _ in params_iter)
                            logger.debug("[Diagnostic] model param count (approx): %d", cnt)
                        except Exception:
                            pass
                        logger.debug("[Diagnostic] model param dtype: %s, device: %s", str(param_dtype), str(param_device))
                else:
                    logger.debug("[Diagnostic] underlying model object has no 'parameters' attribute")
            except Exception as e:
                logger.debug("[Diagnostic] Could not read model parameters dtype/device: %s", str(e))

            # If g_src / g_tgt present, ensure dtype/device match model params
            try:
                for k in ["g_src", "g_tgt", "src_se", "tgt_se", "spk_embed"]:
                    if k in aux_input and isinstance(aux_input[k], torch.Tensor):
                        v = aux_input[k]
                        # if we found model param dtype/device, move tensors to match
                        if param_dtype is not None:
                            if v.dtype != param_dtype:
                                aux_input[k] = v.to(param_dtype)
                                logger.debug("[Diagnostic] Casted aux_input['%s'] to %s", k, str(param_dtype))
                            if param_device is not None and aux_input[k].device != param_device:
                                aux_input[k] = aux_input[k].to(param_device)
                                logger.debug("[Diagnostic] Moved aux_input['%s'] to %s", k, str(param_device))
                        else:
                            # fallback: ensure tensor is float32 on same device as src_spec if available
                            try:
                                target_device = getattr(src_spec, 'device', None)
                                aux_input[k] = aux_input[k].to(torch.float32)
                                if target_device is not None:
                                    aux_input[k] = aux_input[k].to(target_device)
                                    logger.debug("[Diagnostic] Fallback moved aux_input['%s'] to device %s and float32", k, str(target_device))
                            except Exception:
                                pass
            except Exception as e:
                logger.debug("[Diagnostic] Error normalizing aux_input tensors: %s", str(e))

            # Log shapes and stats for known keys
            try:
                for name in ["g_src", "g_tgt", "src_se", "tgt_se", "spk_embed"]:
                    if name in aux_input and isinstance(aux_input[name], torch.Tensor):
                        t = aux_input[name]
                        stats = (float(t.mean()), float(t.std()), float(t.min()), float(t.max()))
                        logger.debug(f"[Diagnostic] {name} shape: {tuple(t.shape)}, dtype: {t.dtype}, stats mean/std/min/max: {stats[0]:.4f}/{stats[1]:.4f}/{stats[2]:.4f}/{stats[3]:.4f}")
            except Exception:
                pass

            # Run inference
            result = model.inference(src_spec, aux_input)

            # If we have src_wave_numpy, compute mse/corr with result (if result is waveform or can be converted)
            try:
                out_wave = None
                if isinstance(result, np.ndarray):
                    out_wave = result
                elif isinstance(result, torch.Tensor):
                    out_wave = result.detach().cpu().numpy()
                elif isinstance(result, dict):
                    # try several common keys
                    for k in ("wav", "audio", "y", "out"):
                        if k in result:
                            val = result[k]
                            if isinstance(val, torch.Tensor):
                                out_wave = val.detach().cpu().numpy()
                                break
                            elif isinstance(val, np.ndarray):
                                out_wave = val
                                break
                    # fallback: check for model_outputs spectrogram/wave tensor
                    if out_wave is None and 'model_outputs' in result:
                        try:
                            val = result['model_outputs']
                            if isinstance(val, torch.Tensor):
                                out_wave = val.detach().cpu().numpy()
                            else:
                                # assume numpy-like or indexable
                                try:
                                    tmp = val[0, 0]
                                    if isinstance(tmp, torch.Tensor):
                                        out_wave = tmp.detach().cpu().numpy()
                                    else:
                                        out_wave = np.array(tmp)
                                except Exception:
                                    out_wave = None
                        except Exception:
                            out_wave = None

                src_arr = None
                if src_wave_numpy is not None:
                    # accept both numpy arrays and torch tensors
                    if isinstance(src_wave_numpy, torch.Tensor):
                        try:
                            src_arr = src_wave_numpy.detach().cpu().numpy()
                        except Exception:
                            src_arr = None
                    else:
                        src_arr = src_wave_numpy

                if out_wave is not None and src_arr is not None:
                    n = min(out_wave.shape[0], src_arr.shape[0])
                    a = src_arr[:n].astype(np.float32)
                    b = out_wave[:n].astype(np.float32)
                    mse = float(np.mean((a - b) ** 2))
                    corr = float(np.corrcoef(a.flatten(), b.flatten())[0, 1]) if n > 1 else float("nan")
                    logger.debug("[Diagnostic] Output vs source -> MSE: %e, Corr: %f", mse, corr)
                else:
                    logger.debug("[Diagnostic] Skipping MSE/corr (missing src_wave or out_wave)")
            except Exception as e:
                logger.debug(f"[Diagnostic] Error computing MSE/corr: {e}")

            return result
        except Exception as e:
            logger.exception(f"[Diagnostic] Inference wrapper failed: {e}")
            # fallback to direct inference to avoid breaking pipeline
            return model.inference(src_spec, aux_input)