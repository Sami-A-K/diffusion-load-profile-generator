"""
In diesem Skript wird die Architektur des Diffusionsmodells definiert, einschließlich der ResBlock-Struktur, des ConditionedDenoiser-Netzwerks und des Diffusionszeitplans.
Es enthält auch Funktionen für die Zeiteinbettung, das Sampling aus dem Diffusionsprozess und den Trainingsschritt.
Diese Implementierung basiert auf der Idee von Denoising Diffusion Probabilistic Models (DDPM) und wurde für die Lastprofilvorhersage abgewandelt.
Diese Modelle stammen ursprünglich aus der Bildgenerierung.
Referenzen dieser Implementierung sind die Werke von:
- Ho et al., "Denoising Diffusion Probabilistic Models" (https://arxiv.org/abs/2006.11239)
- Nichol et al.,Improved Denoising Diffusion Probabilistic Models (https://arxiv.org/abs/2102.09672)
Zu diesen Werken bestehen Implementierungen, von welchen sich diese Implementierung inspirieren ließ.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ResBlock(nn.Module):
    """
    Ein ResBlock, der in der ConditionedDenoiser-Architektur verwendet wird.
    Er besteht aus zwei linearen Schichten mit einer SiLU-Aktivierung.
    Zusätzlich wird ein Film-Mechanismus implementiert.
    Args:
        dim (int): Anzahl der Neuronen
        emb_dim (int): Anzahl der Neuronen für die Konditionsembeddings
    """
    def __init__(self, dim, emb_dim):
        super().__init__()

        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)

        self.emb = nn.Linear(emb_dim, dim * 2)

        self.act = nn.SiLU()

    def forward(self, x, emb):
        h = self.fc1(x)

        scale, shift = self.emb(emb).chunk(2, dim=-1)

        h = h * (1 + scale) + shift #FiLM-Modulation

        h = self.act(h)
        h = self.fc2(h)

        return x + h    # h ist residual

class CrossAttentionBlock(nn.Module):
    """
    Args:
        q_dim (int): Dimension des Query-Inputs (ResBlock hidden_dim).
        kv_dim (int): Dimension von Key/Value-Input (Encoder hidden_dim).
        num_heads (int): Anzahl Attention-Heads.
        dropout (float): Dropout auf den Attention-Output vor dem Residual-Add.
    """
    def __init__(self, q_dim, kv_dim, num_heads=4, dropout=0.0):
        super().__init__()

        self.norm = nn.LayerNorm(q_dim)
        self.attn = nn.MultiheadAttention(embed_dim=q_dim, kdim=kv_dim, vdim=kv_dim, num_heads=num_heads, batch_first=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, context, context_padding_mask=None):
        """
        Args:
            x (torch.FloatTensor): [batch, q_dim], ResBlock-Zustand.
            context (torch.FloatTensor): [batch, seq_len, kv_dim], Encoder-Kontext-Tokens.
            context_padding_mask (torch.BoolTensor, optional): [batch, seq_len], True an Padding-Positionen.
        Returns:
            torch.Tensor: [batch, q_dim], x + Cross-Attention-Update.
        """
        query = self.norm(x).unsqueeze(1)  # [batch, 1, q_dim]

        attn_out, _ = self.attn(query=query, key=context, value=context, key_padding_mask=context_padding_mask, need_weights=False)
        attn_out = attn_out.squeeze(1)  # [batch, q_dim]
        attn_out = self.dropout(attn_out)

        return x + attn_out

class ConditionedDenoiser(nn.Module):
    """
    Res_Net Desnoising-Modell.
    Dieses wird das Rauschen vorhersagen, das aus den Noisy Daten (xt) entfernt werden muss, um die ursprünglichen Daten (x0) zu erhalten.
    Args:
        seq_len (int): Länge der Ausagbe (z.B. 24 für stündliche Daten).
        n_features (int): Anzahl der Merkmale in der Konditionierung
        hidden_dim (int): Anzahl der Neuronen in den verborgenen Schichten.
        time_dim (int): Dimension des Zeit-Embeddings.
        n_res_blocks (int): Anzahl der ResBlocks im Netzwerk.
        cross_attn (bool): Ob ein CrossAttentionBlock genutzt wird
        context_dim (int): Dimension der Encoder-Kontext-Tokens (MeasurementSeriesEncoder-hidden_dim).
    Returns:
        ConditionedDenoiser-Objekt, das die Diffusionsmodell-Architektur implementiert.
    """

    def __init__(self, seq_len=24, n_features=28, hidden_dim=128, time_dim=64, n_res_blocks=4, cross_attn=True, context_dim=128, cross_attn_dropout=0.0):
        super().__init__()

        # Time embedding MLP (sinusoidal embedding)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim))

        # Conditioning embedding
        self.cond_mlp = nn.Sequential(
            nn.Linear(n_features, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim))

        # Input projection
        self.input_proj = nn.Linear(seq_len, hidden_dim)

        self.n_res_blocks = n_res_blocks

        # Cross-attention block
        if cross_attn:
            self.cross_attn_block = CrossAttentionBlock(q_dim=hidden_dim, kv_dim=context_dim, dropout=cross_attn_dropout)
        else:
            self.cross_attn_block = None

        # Residual stack
        self.res_blocks = nn.ModuleList([ResBlock(hidden_dim, time_dim * 2) for i in range(n_res_blocks)])

        # Output projection
        self.output_proj = nn.Linear(hidden_dim, seq_len)

        self.time_dim = time_dim

    def forward(self, xt, t, cond, context=None, context_padding_mask=None):
        # Time embedding, erst sinus dann MLP
        t_emb = diff_timestep_embedding(t, self.time_dim)
        t_emb = self.time_mlp(t_emb)

        # Condition embedding
        cond_emb = self.cond_mlp(cond)

        # Zusammenführen aller Embeddings
        emb = torch.cat([t_emb, cond_emb], dim=-1)

        # Network
        x = self.input_proj(xt)
        if self.cross_attn_block is not None:
            if context is None:
                raise ValueError(
                    "context=None, aber cross_attn=True (Modell wurde mit Cross-Attention-Block "
                    "konstruiert, aber kein Kontext uebergeben) - fuer ein kontextfreies Modell "
                    "cross_attn=False setzen."
                )
            x = self.cross_attn_block(x, context, context_padding_mask)
        for res_block in self.res_blocks:
            x = res_block(x, emb)

        return self.output_proj(x)


class DiffusionSchedule(nn.Module):
    """
    Diffusionszeitplan mit Beta- und Alpha-Werten.
    Args:
        timesteps (int): Anzahl der Diffusionsschritte.
        schedule_type (str): Bisher nur "linear" (Default, Ho et al. 2020)
    Returns:
        DiffusionSchedule-Objekt mit vorab berechneten Beta- und Alpha-Werten.
        Beta-Werte sind bei "linear" fuer 1000 Schritte kalibriert und
        linear zwischen 1e-4 und 0.02 verteilt. Für abweichende Schrittzahlen
        werden beide Endpunkte mit 1000/timesteps skaliert, damit das
        kumulierte Rauschen am Kettenende (alphas_cumprod[-1]) erhalten bleibt.

    Aufruf zB via:
        schedule = DiffusionSchedule(timesteps=500)
    """
    def __init__(self, timesteps=500, schedule_type="linear"): # t: Diffusionsschritte
        super().__init__()
        self.timesteps = timesteps
        self.schedule_type = schedule_type

        if schedule_type == "linear":
            scale = 1000.0 / timesteps
            betas = torch.linspace(scale * 1e-4, scale * 0.02, timesteps)   # beta values, linear schedule.
        # elif schedule_type == "xyz" -> hier können andere Diffusionsfahrpläne hinzugefügt werden
        else:
            raise NotImplementedError(f"schedule_type={schedule_type!r} - nur 'linear' unterstuetzt.")
        self.register_buffer('betas', betas)

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        # Diese Werte werden für die Berechnung von xt und für den Reverse-Diffusionsprozess benötigt, daher werden sie als Buffer gespeichert.
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1.0 - alphas_cumprod))


def diff_timestep_embedding(t, dim):
    """
    Erzeugt ein sinusoidales Embedding für die Zeitindizes.
    Args:
        t (torch.LongTensor): Zeitindizes.
        dim (int): Dimension des Embeddings.
    Returns:
        torch.Tensor: Das sinusoidale Embedding.
    """
    # Sinus embedding
    half = dim // 2

    freqs = torch.exp(-math.log(10000) * torch.arange(0, half, device=t.device) / half)

    args = t[:, None] * freqs[None]

    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)

    if dim % 2 == 1:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb

def q_sample(schedule:DiffusionSchedule, x0, t, noise=None):
    """
    Diffusionsprozess: Erzeugt Noisy Daten (xt) zum Zeitpunkt t aus den ursprünglichen Daten (x0) und einer Rauschkomponente.
    Args:
        schedule (DiffusionSchedule): Diffusionszeitplan.
        x0: Ursprüngliche Daten.
        t: Zeitindizes.
        noise: Rauschkomponente. Wenn None, wird
            standardnormalverteiltes Rauschen generiert.
    Returns:
        torch.Tensor: Noisy data xt zum Zeitpunkt t.

    Via Formel:
    xt = sqt(alphas_cumprod_t) * x0 + sqt(1-alphas_cumprod_t) * noise
    """
    if noise is None:
        noise = torch.randn_like(x0)

    sqrt_ac  = schedule.sqrt_alphas_cumprod[t][:, None]
    sqrt_omac = schedule.sqrt_one_minus_alphas_cumprod[t][:, None]

    return sqrt_ac * x0 + sqrt_omac * noise

def train_step(model: ConditionedDenoiser, schedule: DiffusionSchedule, optimizer: torch.optim.Adam, x0: torch.FloatTensor, cond: torch.FloatTensor, scaler: torch.amp.GradScaler, context: torch.FloatTensor = None, context_padding_mask: torch.BoolTensor = None, amp_dtype: torch.dtype = torch.float16):
    """
    Ein Optimierungsschritt (Training) für das Diffusionsmodell.
    Args:
        model (ConditionedDenoiser): Das zu trainierende Modell.
        schedule (DiffusionSchedule): Der Diffusionszeitplan.
        optimizer (torch.optim.Adam): Der Optimierer für die Modellparameter.
        x0 (torch.FloatTensor): Die ursprünglichen Daten (hier Lastprofile).
        cond (torch.FloatTensor): Die Konditionsvektoren für die Daten.
        scaler (torch.amp.GradScaler): Loss-Scaling für fp16; bei bf16 deaktiviert.
        context (torch.FloatTensor, optional): [batch, seq_len, context_dim], Encoder-Kontext-Tokens.
            None für ein Modell ohne Encoder.
        context_padding_mask (torch.BoolTensor, optional): [batch, seq_len], True an Padding-Positionen.
        amp_dtype (torch.dtype): dtype für torch.autocast (bfloat16 oder float16).
    Returns:
        float: Loss des Batches.
    """
    model.train()
    optimizer.zero_grad()

    batch_size = x0.size(0)
    t = torch.randint(0, schedule.timesteps, (batch_size,),
                      device=x0.device)

    noise = torch.randn_like(x0)    # epsilon, Gausch'sches Rauschen
    xt = q_sample(schedule, x0, t, noise)

    with torch.autocast(device_type=x0.device.type, dtype=amp_dtype, enabled=(x0.device.type == "cuda")):
        noise_pred = model(xt, t, cond, context, context_padding_mask)
        loss = F.mse_loss(noise_pred, noise)

    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)  # Gradient Clipping auf unskalierten Gradienten
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(optimizer)
    scaler.update()
    return loss.item()

@torch.no_grad()
def sample(model: ConditionedDenoiser, schedule: DiffusionSchedule, cond: torch.FloatTensor, device, context: torch.FloatTensor = None, context_padding_mask: torch.BoolTensor = None, x0_min: float | None = -1.0, x0_max: float | None = None):
    """
    Generiert neue Lastprofile durch den Reverse-Diffusionsprozess.
    Args:
        model (ConditionedDenoiser): Das trainierte Modell.
        schedule (DiffusionSchedule): Der Diffusionszeitplan.
        cond (torch.FloatTensor): Die Konditionsvektoren für die Generierung.
        device: Das Gerät, auf dem die Berechnung durchgeführt wird (z.B. 'cuda' oder 'cpu').
        context (torch.FloatTensor, optional): [batch, seq_len, context_dim], Encoder-Kontext-Tokens,
            über alle Diffusionsschritte wiederverwendet. None für ein Modell ohne Encoder.
        context_padding_mask (torch.BoolTensor, optional): [batch, seq_len], True an Padding-Positionen.
        x0_min/x0_max: Clip-Grenzen der x0-Schätzung in jedem Schritt. x0_min=-1 entspricht 0 kWh,
            x0_max ist das Maximum des Trainings-Splits (x0bounds_<version>.json).
    Returns:
        torch.FloatTensor: [batch, 24] auf CPU, im x0-Raum des Trainings (siehe train_model._to_x0).
    """
    model.eval()
    cond = cond.to(device)
    if context is not None:
        context = context.to(device)
    if context_padding_mask is not None:
        context_padding_mask = context_padding_mask.to(device)
    batch = cond.shape[0]
    xt = torch.randn(batch, 24, device=device)

    for i in reversed(range(schedule.timesteps)):
        t = torch.full((batch,), i, dtype=torch.long, device=device)
        pred_noise = model(xt, t, cond, context, context_padding_mask)

        beta_t  = schedule.betas[i]
        alpha_t = 1 - beta_t
        alpha_bar_t = schedule.alphas_cumprod[i]

        if i > 0:
            alpha_bar_t_prev = schedule.alphas_cumprod[i-1]
        else:
            alpha_bar_t_prev = torch.tensor(1.0, device=device)

        # Ab hier findet die berechnung von x_(t-1) statt.
        # Die x0-Schätzung wird geclippt und der Posterior-Mittelwert daraus gebildet
        # (Ho et al. 2020, Gl. 7).
        x0_pred = (xt - torch.sqrt(1 - alpha_bar_t) * pred_noise) / torch.sqrt(alpha_bar_t)
        x0_pred = x0_pred.clamp(min=x0_min, max=x0_max)

        mean = (
            (torch.sqrt(alpha_bar_t_prev) * beta_t) / (1 - alpha_bar_t) * x0_pred
            + (torch.sqrt(alpha_t) * (1 - alpha_bar_t_prev)) / (1 - alpha_bar_t) * xt
        )

        if i > 0:
            noise   = torch.randn_like(xt)
            var = beta_t * (1-alpha_bar_t_prev)/(1-alpha_bar_t)
            xt = mean + torch.sqrt(var) * noise
        else:
            xt = mean

    return xt.cpu()
