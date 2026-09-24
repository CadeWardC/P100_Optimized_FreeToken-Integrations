#include "core.h"
#include <httplib.h>
#include <FL/Fl.H>
#include <FL/Fl_Double_Window.H>
#include <FL/Fl_Group.H>
#include <FL/Fl_Scroll.H>
#include <FL/Fl_Input.H>
#include <FL/Fl_Multiline_Input.H>
#include <FL/Fl_Check_Button.H>
#include <FL/Fl_Button.H>
#include <FL/Fl_Box.H>
#include <FL/Fl_Text_Display.H>
#include <FL/Fl_Text_Buffer.H>
#include <FL/Fl_Choice.H>
#include <FL/Fl_Hold_Browser.H>
#include <FL/Fl_Value_Slider.H>
#include <FL/Fl_RGB_Image.H>
#include <FL/Fl_Image_Surface.H>
#include <FL/Fl_PNG_Image.H>
#include <FL/Fl_Progress.H>
#include <FL/Fl_Native_File_Chooser.H>
#include <FL/Fl_Menu_Button.H>
#include <FL/fl_ask.H>
#include <FL/fl_draw.H>
#ifdef _WIN32
#include <FL/platform.H>
#include <dwmapi.h>
#endif
#include <array>
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <atomic>
#include <fstream>
#include <map>
#include <mutex>
#include <thread>
#include <deque>
#include <chrono>
#include <vector>
#include <functional>

namespace theme {
const Fl_Color background         = fl_rgb_color(7, 10, 18);    // #070a12 midnight workspace
const Fl_Color sidebar            = fl_rgb_color(8, 11, 18);    // #080b12 navigation rail
const Fl_Color panel              = fl_rgb_color(15, 21, 34);   // #0f1522 card surface
const Fl_Color panel_elevated     = fl_rgb_color(20, 29, 47);   // #141d2f elevated surface
const Fl_Color border             = fl_rgb_color(35, 48, 72);   // #233048 crisp border
const Fl_Color border_subtle      = fl_rgb_color(24, 33, 51);   // #182133
const Fl_Color text               = fl_rgb_color(225, 233, 246);// #e1e9f6 primary text
const Fl_Color text_bright        = fl_rgb_color(255, 255, 255);
const Fl_Color muted              = fl_rgb_color(145, 160, 184);// #91a0b8 secondary text
const Fl_Color subtle             = fl_rgb_color(80, 95, 120);  // #505f78 dim hints
const Fl_Color accent             = fl_rgb_color(76, 216, 255); // #4cd8ff signal cyan
const Fl_Color accent_dark        = fl_rgb_color(11, 42, 58);   // #0b2a3a
const Fl_Color accent_hover       = fl_rgb_color(126, 228, 255);
const Fl_Color violet             = fl_rgb_color(153, 132, 255);// #9984ff secondary signal
const Fl_Color bubble_user        = fl_rgb_color(18, 31, 49);   // #121f31 user message bubble
const Fl_Color bubble_ai          = fl_rgb_color(13, 19, 31);   // #0d131f assistant message bubble
const Fl_Color status_green       = fl_rgb_color(74, 222, 128); // #4ade80
const Fl_Color status_amber       = fl_rgb_color(251, 191, 36); // #fbbf24
const Fl_Color status_red         = fl_rgb_color(248, 113, 113);// #f87171
const Fl_Color status_blue        = fl_rgb_color(96, 165, 250); // #60a5fa

constexpr Fl_Boxtype rounded = FL_FREE_BOXTYPE;

void surface(int x, int y, int w, int h, Fl_Color c) {
    fl_color(c); fl_rounded_rectf(x, y, w, h, 9);
    fl_color(border); fl_rounded_rect(x, y, w, h, 8);
}
void card(int x, int y, int w, int h, Fl_Color bg = panel, Fl_Color bdr = border, int radius = 8) {
    fl_color(bg); fl_rounded_rectf(x, y, w, h, radius);
    fl_color(bdr); fl_rounded_rect(x, y, w, h, radius);
}
void style(Fl_Group* group) {
    for(int i = 0; i < group->children(); ++i) {
        auto* w = group->child(i);
        if(!dynamic_cast<Fl_Box*>(w)) w->labelsize(13);
        if(w->labelcolor() != muted) w->labelcolor(text);
        if(auto* g = dynamic_cast<Fl_Group*>(w)) {
            if(g->box() == FL_NO_BOX) { g->box(FL_FLAT_BOX); g->color(background); }
            style(g);
        }
        if(auto* b = dynamic_cast<Fl_Button*>(w)) { b->box(rounded); b->down_box(rounded); b->color(panel); b->selection_color(border); }
        if(auto* c = dynamic_cast<Fl_Check_Button*>(w)) { c->box(FL_NO_BOX); c->down_box(FL_FLAT_BOX); c->selection_color(accent); }
        if(auto* f = dynamic_cast<Fl_Input*>(w)) { f->box(rounded); f->color(panel); f->textcolor(text); f->cursor_color(accent); f->selection_color(fl_rgb_color(45,72,65)); f->textsize(13); }
        if(auto* d = dynamic_cast<Fl_Text_Display*>(w)) { d->box(rounded); d->color(panel); d->textcolor(text); d->selection_color(border); }
        if(auto* c = dynamic_cast<Fl_Choice*>(w)) { c->box(rounded); c->down_box(rounded); c->color(panel); c->textcolor(text); c->textsize(13); c->selection_color(border); }
        if(auto* b = dynamic_cast<Fl_Hold_Browser*>(w)) { b->box(rounded); b->color(panel); b->textcolor(text); b->textsize(14); b->selection_color(fl_rgb_color(36,54,48)); }
        if(auto* s = dynamic_cast<Fl_Value_Slider*>(w)) { s->box(FL_FLAT_BOX); s->color(panel); s->selection_color(accent); s->textcolor(text); s->textsize(13); }
        if(auto* s = dynamic_cast<Fl_Scroll*>(w)) { s->scrollbar.box(FL_FLAT_BOX); s->scrollbar.slider(FL_FLAT_BOX); s->scrollbar.color(background); s->scrollbar.selection_color(border); }
    }
}
}

// Custom modern sidebar navigation button with active left pill
class NavButton : public Fl_Button {
public:
    bool active = false;
    const char* badge = nullptr;
    const char* icon_glyph = nullptr;
    NavButton(int x, int y, int w, int h, const char* label, const char* icon = nullptr)
        : Fl_Button(x, y, w, h, label), icon_glyph(icon) {
        box(FL_NO_BOX);
    }
    void draw() override {
        if (active) {
            fl_color(theme::panel_elevated);
            fl_rounded_rectf(x(), y(), w(), h(), 8);
            fl_color(theme::accent);
            fl_rounded_rectf(x(), y() + 8, 3, h() - 16, 2);
        } else if (Fl::belowmouse() == this) {
            fl_color(theme::panel);
            fl_rounded_rectf(x(), y(), w(), h(), 6);
        } else {
            fl_color(theme::sidebar);
            fl_rectf(x(), y(), w(), h());
        }
        int tx = x() + 14;
        if (icon_glyph && icon_glyph[0]) {
            fl_color(active ? theme::accent : theme::muted);
            fl_font(FL_COURIER_BOLD, 10);
            fl_draw(icon_glyph, tx, y(), 24, h(), FL_ALIGN_LEFT);
            tx += 30;
        }
        fl_color(active ? theme::text_bright : theme::muted);
        fl_font(active ? FL_HELVETICA_BOLD : FL_HELVETICA, 13);
        fl_draw(label() ? label() : "", tx, y(), w() - (tx - x()) - (badge ? 34 : 10), h(), FL_ALIGN_LEFT);
        if (badge && badge[0]) {
            int bw = 24, bh = 18;
            int bx = x() + w() - bw - 10, by = y() + (h() - bh) / 2;
            fl_color(theme::border);
            fl_rounded_rectf(bx, by, bw, bh, 9);
            fl_color(theme::text);
            fl_font(FL_HELVETICA_BOLD, 11);
            fl_draw(badge, bx, by, bw, bh, FL_ALIGN_CENTER);
        }
    }
};

// Polished action button
class ActionBtn : public Fl_Button {
public:
    enum Style { Primary, Secondary, Danger, Ghost };
    Style btn_style = Secondary;
    ActionBtn(int x, int y, int w, int h, const char* label = 0, Style st = Secondary)
        : Fl_Button(x, y, w, h, label), btn_style(st) {}
    void draw() override {
        Fl_Color bg, fg, bdr;
        if (!active_r()) {
            bg = theme::panel; fg = theme::subtle; bdr = theme::border_subtle;
        } else if (btn_style == Primary) {
            bg = Fl::belowmouse() == this ? theme::accent_hover : theme::accent;
            fg = fl_rgb_color(10, 16, 13);
            bdr = bg;
        } else if (btn_style == Danger) {
            bg = Fl::belowmouse() == this ? fl_rgb_color(80, 24, 24) : fl_rgb_color(52, 20, 20);
            fg = theme::status_red;
            bdr = theme::status_red;
        } else if (btn_style == Ghost) {
            bg = Fl::belowmouse() == this ? theme::panel_elevated : theme::background;
            fg = theme::muted;
            bdr = theme::border_subtle;
        } else {
            bg = Fl::belowmouse() == this ? theme::panel_elevated : theme::panel;
            fg = theme::text;
            bdr = theme::border;
        }
        fl_color(bg); fl_rounded_rectf(x(), y(), w(), h(), 8);
        fl_color(bdr); fl_rounded_rect(x(), y(), w(), h(), 8);
        fl_color(fg);
        fl_font(btn_style == Primary ? FL_HELVETICA_BOLD : FL_HELVETICA, labelsize() ? labelsize() : 13);
        fl_draw(label() ? label() : "", x(), y(), w(), h(), FL_ALIGN_CENTER);
    }
};

class Switch : public Fl_Check_Button {
    void draw() override {
        fl_color(parent()->color()); fl_rectf(x(), y(), w(), h());
        fl_color(value() ? theme::accent : theme::border);
        fl_rounded_rectf(x() + w() - 44, y() + 6, 38, 20, 10);
        fl_color(value() ? theme::sidebar : theme::muted);
        fl_pie(x() + w() - 42 + (value() ? 18 : 0), y() + 8, 16, 16, 0, 360);
        fl_color(active_r() ? theme::text : theme::muted);
        fl_font(FL_HELVETICA, 13);
        fl_draw(label() ? label() : "", x() + 4, y(), w() - 54, h(), FL_ALIGN_LEFT);
    }
public:
    using Fl_Check_Button::Fl_Check_Button;
};

class Choice : public Fl_Choice {
    void draw() override {
        theme::surface(x(), y(), w(), h(), theme::panel);
        fl_color(active_r() ? theme::text : theme::muted);
        fl_font(FL_HELVETICA, 13);
        fl_push_clip(x() + 10, y(), w() - 36, h());
        fl_draw(mvalue() ? mvalue()->label() : "Select", x() + 10, y(), w() - 36, h(), FL_ALIGN_LEFT);
        fl_pop_clip();
        fl_color(theme::muted);
        fl_line(x() + w() - 20, y() + h() / 2 - 2, x() + w() - 16, y() + h() / 2 + 2, x() + w() - 12, y() + h() / 2 - 2);
    }
public:
    using Fl_Choice::Fl_Choice;
};

class Disclosure : public Fl_Button {
    void draw() override {
        theme::surface(x(), y(), w(), h(), theme::panel);
        fl_color(theme::text);
        fl_font(FL_HELVETICA_BOLD, 13);
        fl_draw(label(), x() + 14, y(), w() - 48, h(), FL_ALIGN_LEFT);
        int cx = x() + w() - 20, cy = y() + h() / 2;
        fl_color(theme::muted);
        if(expanded) fl_line(cx - 4, cy - 2, cx, cy + 2, cx + 4, cy - 2);
        else fl_line(cx - 2, cy - 4, cx + 2, cy, cx - 2, cy + 4);
    }
public:
    bool expanded = false;
    using Fl_Button::Fl_Button;
};

class Slider : public Fl_Value_Slider {
    void draw() override {
        fl_color(parent()->color()); fl_rectf(x(), y(), w(), h());
        char text[128]; format(text);
        fl_color(theme::text); fl_font(FL_HELVETICA, 13);
        fl_draw(all_at_max && value() == maximum() ? "All" : text, x(), y(), value_width() - 4, h(), FL_ALIGN_CENTER);
        int start = x() + value_width() + 6, end = x() + w() - 6, cy = y() + h() / 2;
        double ratio = maximum() > minimum() ? (value() - minimum()) / (maximum() - minimum()) : 0;
        int knob = start + static_cast<int>((end - start) * std::clamp(ratio, 0.0, 1.0));
        fl_color(theme::border);
        fl_rounded_rectf(start, cy - 3, end - start, 6, 3);
        fl_color(theme::accent);
        if(knob > start) fl_rounded_rectf(start, cy - 3, knob - start, 6, 3);
        fl_color(Fl::belowmouse() == this ? theme::accent_hover : theme::accent);
        fl_pie(knob - 7, cy - 7, 14, 14, 0, 360);
    }
public:
    bool all_at_max = false;
    using Fl_Value_Slider::Fl_Value_Slider;
};

// Modal loading card during model startup
class LoadingCard : public Fl_Group {
public:
    Fl_Box* title_box = nullptr;
    Fl_Box* stage_box = nullptr;
    Fl_Progress* progress_bar = nullptr;
    Fl_Box* steps_box = nullptr;
    ActionBtn* cancel_btn = nullptr;

    LoadingCard(int x, int y, int w, int h) : Fl_Group(x, y, w, h) {
        box(FL_NO_BOX);
        title_box = new Fl_Box(x + 20, y + 16, w - 160, 24, "Loading Model...");
        title_box->align(FL_ALIGN_LEFT | FL_ALIGN_INSIDE);
        title_box->labelfont(FL_HELVETICA_BOLD);
        title_box->labelsize(15);
        title_box->labelcolor(theme::text_bright);

        stage_box = new Fl_Box(x + 20, y + 42, w - 40, 20, "Initializing backend server...");
        stage_box->align(FL_ALIGN_LEFT | FL_ALIGN_INSIDE);
        stage_box->labelsize(12);
        stage_box->labelcolor(theme::accent);

        progress_bar = new Fl_Progress(x + 20, y + 68, w - 40, 10);
        progress_bar->box(FL_FLAT_BOX);
        progress_bar->color(theme::panel);
        progress_bar->selection_color(theme::accent);
        progress_bar->minimum(0); progress_bar->maximum(100); progress_bar->value(5);

        steps_box = new Fl_Box(x + 20, y + 88, w - 40, 20, "1. Read File (5%)  ->  2. Tensors (15%)  ->  3. GPU Offload (35%)  ->  4. Context (70%)  ->  5. API Ready (95%)");
        steps_box->align(FL_ALIGN_LEFT | FL_ALIGN_INSIDE);
        steps_box->labelsize(11);
        steps_box->labelcolor(theme::muted);

        cancel_btn = new ActionBtn(x + w - 120, y + 16, 100, 30, "Stop Startup", ActionBtn::Danger);
        cancel_btn->labelsize(11);
        end();
    }
    void draw() override {
        theme::card(x(), y(), w(), h(), theme::panel_elevated, theme::accent, 10);
        Fl_Group::draw();
    }
};

// LM Studio style centered model capsule in the header
class ModelCapsule : public Fl_Group {
public:
    Fl_Button* model_btn = nullptr;
    Fl_Box* fit_chip = nullptr;
    Fl_Box* status_dot = nullptr;
    ActionBtn* action_btn = nullptr;

    ModelCapsule(int X, int Y, int W, int H) : Fl_Group(X, Y, W, H) {
        box(FL_NO_BOX);

        model_btn = new Fl_Button(X + 12, Y + 3, W - 232, H - 6, "Select a model to load... ▼");
        model_btn->box(FL_NO_BOX);
        model_btn->labelcolor(theme::text_bright);
        model_btn->labelsize(13);
        model_btn->labelfont(FL_HELVETICA_BOLD);
        model_btn->align(FL_ALIGN_LEFT | FL_ALIGN_INSIDE);

        fit_chip = new Fl_Box(X + W - 216, Y + 5, 78, H - 10, "0/30 GPU");
        fit_chip->align(FL_ALIGN_CENTER | FL_ALIGN_INSIDE);
        fit_chip->labelsize(10);
        fit_chip->labelcolor(theme::muted);

        status_dot = new Fl_Box(X + W - 134, Y + 5, 54, H - 10, "○ Idle");
        status_dot->align(FL_ALIGN_CENTER | FL_ALIGN_INSIDE);
        status_dot->labelsize(11);
        status_dot->labelcolor(theme::muted);

        action_btn = new ActionBtn(X + W - 76, Y + 5, 68, H - 10, "Load", ActionBtn::Primary);
        action_btn->labelsize(11);

        end();
    }

    void update(const std::string& model_name, const std::string& fit, bool is_ready, bool is_busy, int pct = 0) {
        if(model_name.empty()) {
            model_btn->copy_label("Select a model to load... ▼");
            fit_chip->copy_label("");
            status_dot->copy_label("○ Idle");
            status_dot->labelcolor(theme::muted);
            action_btn->copy_label("Load");
            action_btn->btn_style = ActionBtn::Primary;
            action_btn->deactivate();
        } else {
            model_btn->copy_label(("✦ " + model_name + " ▼").c_str());
            fit_chip->copy_label(fit.c_str());
            if(is_busy && !is_ready) {
                char b[32]; snprintf(b, sizeof b, "⏳ %d%%", std::max(5, pct));
                status_dot->copy_label(b);
                status_dot->labelcolor(theme::status_amber);
                action_btn->copy_label("Stop");
                action_btn->btn_style = ActionBtn::Danger;
                action_btn->activate();
            } else if(is_ready) {
                status_dot->copy_label("● Ready");
                status_dot->labelcolor(theme::status_green);
                action_btn->copy_label("Unload");
                action_btn->btn_style = ActionBtn::Secondary;
                action_btn->activate();
            } else {
                status_dot->copy_label("○ Stopped");
                status_dot->labelcolor(theme::muted);
                action_btn->copy_label("Load");
                action_btn->btn_style = ActionBtn::Primary;
                action_btn->activate();
            }
        }
        redraw();
    }

    void draw() override {
        fl_color(fl_rgb_color(14, 21, 34));
        fl_rounded_rectf(x(), y(), w(), h(), h() / 2);
        fl_color(theme::border);
        fl_rounded_rect(x(), y(), w(), h(), h() / 2);

        fl_color(theme::border_subtle);
        fl_line(x() + w() - 222, y() + 6, x() + w() - 222, y() + h() - 6);

        fl_color(theme::accent);
        fl_pie(x() + 5, y() + h() / 2 - 2, 4, 4, 0, 360);

        Fl_Group::draw();
    }
};

// LM Studio style individual message card bubble
class ChatMessageCard : public Fl_Group {
public:
    int role = 0; // 0 = user, 1 = assistant
    std::string author;
    std::string model_name;
    std::string content;
    std::string reasoning;
    std::string stats;
    bool is_streaming = false;
    ActionBtn* copy_btn = nullptr;
    int content_h = 0;
    int reasoning_h = 0;

    ChatMessageCard(int X, int Y, int W, int R, const std::string& auth, const std::string& mod,
                    const std::string& cont, const std::string& st = "", const std::string& reas = "")
        : Fl_Group(X, Y, W, 80), role(R), author(auth), model_name(mod), content(cont), stats(st), reasoning(reas) {
        box(FL_NO_BOX);

        copy_btn = new ActionBtn(X + W - 92, Y + 10, 78, 24, "Copy", ActionBtn::Secondary);
        copy_btn->labelsize(11);
        copy_btn->callback([](Fl_Widget* w, void* p) {
            auto* card = static_cast<ChatMessageCard*>(p);
            Fl::copy(card->content.c_str(), static_cast<int>(card->content.size()), 1);
            card->copy_btn->copy_label("Copied!");
            card->copy_btn->redraw();
            Fl::add_timeout(1.5, [](void* p) {
                auto* b = static_cast<ActionBtn*>(p);
                b->copy_label("Copy");
                b->redraw();
            }, card->copy_btn);
        }, this);
        if(role == 0) copy_btn->hide();

        end();
        recalc_size();
    }

    void recalc_size() {
        int text_w = w() - 36;
        if(text_w < 100) text_w = 100;

        fl_font(FL_HELVETICA, 13);
        int mw = text_w, mh = 0;
        fl_measure(content.empty() ? " " : content.c_str(), mw, mh, 0);
        content_h = std::max(20, mh);

        if(!reasoning.empty()) {
            fl_font(FL_HELVETICA_ITALIC, 12);
            int rw = text_w - 24, rh = 0;
            fl_measure(reasoning.c_str(), rw, rh, 0);
            reasoning_h = std::max(20, rh) + 32;
        } else {
            reasoning_h = 0;
        }

        int total_h = 0;
        if(role == 0) {
            total_h = 14 + 18 + 8 + content_h + 16;
        } else {
            total_h = 14 + 22 + 10 + reasoning_h + (reasoning_h ? 8 : 0) + content_h + 14 + 30 + 12;
        }
        size(w(), total_h);
        if(copy_btn && role == 1) {
            copy_btn->position(x() + w() - 92, y() + total_h - 36);
        }
    }

    void append_token(const std::string& token) {
        content += token;
        recalc_size();
        redraw();
    }

    void append_reasoning(const std::string& text) {
        reasoning += text;
        recalc_size();
        redraw();
    }

    void set_stats_line(const std::string& st) {
        stats = st;
        is_streaming = false;
        recalc_size();
        redraw();
    }

    void draw() override {
        int card_x = x() + 4;
        int card_y = y() + 2;
        int card_w = w() - 8;
        int card_h = h() - 4;

        if(role == 0) {
            // User Message Card
            fl_color(theme::bubble_user);
            fl_rounded_rectf(card_x, card_y, card_w, card_h, 12);
            fl_color(fl_rgb_color(37, 58, 82));
            fl_rounded_rect(card_x, card_y, card_w, card_h, 12);

            // User Tag
            fl_color(theme::accent);
            fl_font(FL_HELVETICA_BOLD, 12);
            fl_draw("YOU", card_x + 16, card_y + 12, 100, 16, FL_ALIGN_LEFT);

            // Content
            fl_color(theme::text_bright);
            fl_font(FL_HELVETICA, 13);
            fl_draw(content.c_str(), card_x + 16, card_y + 34, card_w - 32, content_h, FL_ALIGN_LEFT | FL_ALIGN_TOP | FL_ALIGN_WRAP);
        } else {
            // Assistant Message Card
            fl_color(theme::bubble_ai);
            fl_rounded_rectf(card_x, card_y, card_w, card_h, 12);
            fl_color(theme::border_subtle);
            fl_rounded_rect(card_x, card_y, card_w, card_h, 12);

            // Signal avatar
            int av_x = card_x + 16, av_y = card_y + 12;
            fl_color(theme::accent_dark);
            fl_rounded_rectf(av_x, av_y, 22, 22, 6);
            fl_color(theme::accent);
            fl_font(FL_HELVETICA_BOLD, 13);
            fl_draw("F", av_x, av_y, 22, 22, FL_ALIGN_CENTER);

            // Header Title
            fl_color(theme::accent);
            fl_font(FL_HELVETICA_BOLD, 12);
            std::string hdr = "ASSISTANT  ·  " + (model_name.empty() ? "Local Model" : model_name);
            fl_draw(hdr.c_str(), av_x + 30, av_y + 2, card_w - 160, 18, FL_ALIGN_LEFT);

            fl_color(theme::muted);
            fl_font(FL_HELVETICA, 11);
            fl_draw("Local Engine", card_x + card_w - 100, av_y + 2, 84, 18, FL_ALIGN_RIGHT);

            int cur_y = card_y + 42;

            // Thinking box
            if(!reasoning.empty()) {
                int r_w = card_w - 32;
                int r_h = reasoning_h;
                fl_color(fl_rgb_color(22, 19, 38));
                fl_rounded_rectf(card_x + 16, cur_y, r_w, r_h, 8);
                fl_color(fl_rgb_color(58, 48, 80));
                fl_rounded_rect(card_x + 16, cur_y, r_w, r_h, 8);

                fl_color(theme::violet);
                fl_font(FL_HELVETICA_BOLD, 11);
                fl_draw("THINKING PROCESS", card_x + 26, cur_y + 8, r_w - 20, 14, FL_ALIGN_LEFT);

                fl_color(fl_rgb_color(160, 170, 185));
                fl_font(FL_HELVETICA_ITALIC, 12);
                fl_draw(reasoning.c_str(), card_x + 26, cur_y + 26, r_w - 20, r_h - 32, FL_ALIGN_LEFT | FL_ALIGN_TOP | FL_ALIGN_WRAP);
                cur_y += r_h + 10;
            }

            // Body Content
            fl_color(theme::text);
            fl_font(FL_HELVETICA, 13);
            fl_draw(content.c_str(), card_x + 16, cur_y, card_w - 32, content_h, FL_ALIGN_LEFT | FL_ALIGN_TOP | FL_ALIGN_WRAP);

            // Footer pills
            int foot_y = card_y + card_h - 34;
            if(!stats.empty()) {
                int pill_x = card_x + 16;
                int pill_w = card_w - 120;
                fl_color(fl_rgb_color(9, 16, 27));
                fl_rounded_rectf(pill_x, foot_y, pill_w, 24, 6);
                fl_color(fl_rgb_color(30, 40, 56));
                fl_rounded_rect(pill_x, foot_y, pill_w, 24, 6);

                fl_color(theme::accent);
                fl_font(FL_HELVETICA_BOLD, 11);
                fl_draw(stats.c_str(), pill_x + 10, foot_y, pill_w - 20, 24, FL_ALIGN_LEFT | FL_ALIGN_INSIDE);
            }
        }
        Fl_Group::draw();
    }
};

// LM Studio style scrollable chat stream
class ChatDisplay : public Fl_Scroll {
public:
    std::vector<ChatMessageCard*> cards;
    ChatMessageCard* active_assistant_card = nullptr;
    int content_bottom = 12;

    ChatDisplay(int x, int y, int w, int h) : Fl_Scroll(x, y, w, h) {
        type(Fl_Scroll::VERTICAL);
        box(FL_FLAT_BOX);
        color(theme::background);
        scrollbar.box(FL_FLAT_BOX);
        scrollbar.color(theme::background);
        scrollbar.slider(FL_FLAT_BOX);
        scrollbar.selection_color(theme::border);
        end();
    }

    void auto_scroll() {
        init_sizes();
        int target_y = std::max(0, content_bottom - h() + 20);
        scroll_to(0, target_y);
    }

    void clear_messages() {
        for(auto* c : cards) {
            remove(c);
            delete c;
        }
        cards.clear();
        active_assistant_card = nullptr;
        content_bottom = 12;
        init_sizes();
        scroll_to(0, 0);
        redraw();
    }

    void add_user_message(const std::string& text) {
        begin();
        int cw = w() - 24;
        if(scrollbar.visible()) cw -= scrollbar.w();
        auto* card = new ChatMessageCard(x() + 10, y() + content_bottom, cw, 0, "You", "", text);
        end();
        cards.push_back(card);
        content_bottom += card->h() + 12;
        auto_scroll();
        redraw();
    }

    void start_assistant_message(const std::string& model_name) {
        begin();
        int cw = w() - 24;
        if(scrollbar.visible()) cw -= scrollbar.w();
        auto* card = new ChatMessageCard(x() + 10, y() + content_bottom, cw, 1, "Assistant", model_name, "");
        card->is_streaming = true;
        end();
        cards.push_back(card);
        active_assistant_card = card;
        content_bottom += card->h() + 12;
        auto_scroll();
        redraw();
    }

    void append_token(const std::string& token) {
        if(!active_assistant_card) return;
        int old_h = active_assistant_card->h();
        active_assistant_card->append_token(token);
        int diff = active_assistant_card->h() - old_h;
        if(diff != 0) {
            content_bottom += diff;
            auto_scroll();
        }
        active_assistant_card->redraw();
    }

    void append_reasoning(const std::string& text) {
        if(!active_assistant_card) return;
        int old_h = active_assistant_card->h();
        active_assistant_card->append_reasoning(text);
        int diff = active_assistant_card->h() - old_h;
        if(diff != 0) {
            content_bottom += diff;
            auto_scroll();
        }
        active_assistant_card->redraw();
    }

    void finish_assistant_message(const std::string& stats) {
        if(!active_assistant_card) return;
        int old_h = active_assistant_card->h();
        active_assistant_card->set_stats_line(stats);
        int diff = active_assistant_card->h() - old_h;
        if(diff != 0) {
            content_bottom += diff;
            auto_scroll();
        }
        active_assistant_card = nullptr;
        redraw();
    }

    void resize(int X, int Y, int W, int H) override {
        Fl_Scroll::resize(X, Y, W, H);
        int cw = W - 24;
        if(scrollbar.visible()) cw -= scrollbar.w();
        int cy = 12;
        for(auto* c : cards) {
            c->position(X + 10, Y + cy);
            c->size(cw, c->h());
            c->recalc_size();
            cy += c->h() + 12;
        }
        content_bottom = cy;
        init_sizes();
    }

    void draw() override {
        Fl_Scroll::draw();
        if(cards.empty()) {
            int cx = x();
            int cy = y() + h() / 2 - 142;
            int cw = w();

            fl_color(theme::accent_dark);
            fl_rounded_rectf(cx + cw / 2 - 28, cy, 56, 56, 16);
            fl_color(theme::accent);
            fl_font(FL_COURIER_BOLD, 18);
            fl_draw("FT", cx, cy, cw, 56, FL_ALIGN_CENTER);

            fl_color(theme::text_bright);
            fl_font(FL_HELVETICA_BOLD, 24);
            fl_draw("Your local inference workspace", cx, cy + 70, cw, 30, FL_ALIGN_CENTER);

            fl_color(theme::muted);
            fl_font(FL_HELVETICA, 13);
            fl_draw("Private, hardware-aware, and tuned for the Tesla P100.", cx, cy + 104, cw, 22, FL_ALIGN_CENTER);

            const char* titles[] = {"01  CHOOSE A MODEL", "02  CHECK THE FIT", "03  START A SESSION"};
            const char* notes[] = {"Pick a GGUF from Model Vault", "Review VRAM and offload plan", "Load once, then ask anything"};
            int card_w = std::min(220, (cw - 72) / 3);
            int total_w = card_w * 3 + 20 * 2;
            int sx = cx + (cw - total_w) / 2;
            for(int i = 0; i < 3; ++i) {
                int bx = sx + i * (card_w + 20), by = cy + 148;
                fl_color(theme::panel);
                fl_rounded_rectf(bx, by, card_w, 72, 10);
                fl_color(i == 0 ? theme::accent : theme::border);
                fl_rounded_rect(bx, by, card_w, 72, 10);
                fl_color(i == 0 ? theme::accent : theme::text);
                fl_font(FL_COURIER_BOLD, 10);
                fl_draw(titles[i], bx + 14, by + 10, card_w - 28, 18, FL_ALIGN_LEFT);
                fl_color(theme::muted);
                fl_font(FL_HELVETICA, 11);
                fl_draw(notes[i], bx + 14, by + 34, card_w - 28, 28, FL_ALIGN_LEFT | FL_ALIGN_WRAP);
            }
        }
    }

    // Compatibility dummy methods for legacy Fl_Text_Display calls
    void buffer(Fl_Text_Buffer*) {}
    void highlight_data(Fl_Text_Buffer*, const Fl_Text_Display::Style_Table_Entry*, int, char, Fl_Text_Display::Unfinished_Style_Cb, void*) {}
    void wrap_mode(int, int) {}
    void insert_position(int) {}
    void show_insert_position() { auto_scroll(); }
};

// Enter-to-send multiline input field with placeholder
class ChatInput : public Fl_Multiline_Input {
public:
    std::function<void()> on_submit;
    ChatInput(int x, int y, int w, int h) : Fl_Multiline_Input(x, y, w, h) {
        box(FL_NO_BOX);
        color(fl_rgb_color(15, 23, 38));
        textcolor(theme::text_bright);
        cursor_color(theme::accent);
        textsize(13);
        wrap(1);
    }
    int handle(int e) override {
        if(e == FL_KEYDOWN) {
            if(Fl::event_key() == FL_Enter && !Fl::event_state(FL_SHIFT) && !Fl::event_state(FL_CTRL)) {
                if(on_submit) on_submit();
                return 1;
            }
        }
        return Fl_Multiline_Input::handle(e);
    }
    void draw() override {
        Fl_Multiline_Input::draw();
        if(strlen(value()) == 0 && Fl::focus() != this) {
            fl_color(theme::subtle);
            fl_font(FL_HELVETICA, 13);
            fl_draw("Ask your local model anything...", x() + 6, y() + 4, w() - 12, 22, FL_ALIGN_LEFT);
        }
    }
};

// Floating elevated composer card at the bottom of the chat workspace
class ComposerCard : public Fl_Group {
public:
    ChatInput* input = nullptr;
    ActionBtn* new_chat_btn = nullptr;
    ActionBtn* send_btn = nullptr;
    ActionBtn* stop_btn = nullptr;
    Fl_Box* ctx_chip = nullptr;
    Fl_Box* speed_chip = nullptr;

    ComposerCard(int X, int Y, int W, int H) : Fl_Group(X, Y, W, H) {
        box(FL_NO_BOX);

        input = new ChatInput(X + 14, Y + 10, W - 28, 44);

        new_chat_btn = new ActionBtn(X + 14, Y + 60, 104, 28, "+ New session", ActionBtn::Secondary);
        new_chat_btn->labelsize(11);

        ctx_chip = new Fl_Box(X + 126, Y + 60, 160, 28, "ctx 0 / 4096 tokens");
        ctx_chip->align(FL_ALIGN_LEFT | FL_ALIGN_INSIDE);
        ctx_chip->labelsize(11);
        ctx_chip->labelcolor(theme::muted);

        speed_chip = new Fl_Box(X + 296, Y + 60, 180, 28, "");
        speed_chip->align(FL_ALIGN_LEFT | FL_ALIGN_INSIDE);
        speed_chip->labelsize(11);
        speed_chip->labelcolor(theme::accent);
        speed_chip->hide();

        stop_btn = new ActionBtn(X + W - 186, Y + 60, 80, 28, "Stop ■", ActionBtn::Danger);
        stop_btn->labelsize(11);
        stop_btn->hide();

        send_btn = new ActionBtn(X + W - 98, Y + 60, 84, 28, "Send  ↗", ActionBtn::Primary);
        send_btn->labelsize(11);

        end();
    }

    void resize(int X, int Y, int W, int H) override {
        Fl_Group::resize(X, Y, W, H);
        if(input) input->resize(X + 14, Y + 10, W - 28, 44);
        if(new_chat_btn) new_chat_btn->resize(X + 14, Y + 60, 104, 28);
        if(ctx_chip) ctx_chip->resize(X + 126, Y + 60, 160, 28);
        if(speed_chip) speed_chip->resize(X + 296, Y + 60, 180, 28);
        if(stop_btn) stop_btn->resize(X + W - 186, Y + 60, 80, 28);
        if(send_btn) send_btn->resize(X + W - 98, Y + 60, 84, 28);
    }

    void draw() override {
        fl_color(fl_rgb_color(13, 20, 33));
        fl_rounded_rectf(x(), y(), w(), h(), 12);
        fl_color(theme::border);
        fl_rounded_rect(x(), y(), w(), h(), 12);

        fl_color(theme::border_subtle);
        fl_line(x() + 12, y() + 54, x() + w() - 12, y() + 54);

        Fl_Group::draw();
    }
};

// LM Studio style in-chat right rail: parameters & system prompt inspector
class ChatInspector : public Fl_Group {
public:
    Choice* preset_choice = nullptr;
    Fl_Multiline_Input* sys_prompt = nullptr;
    ActionBtn* apply_sys_btn = nullptr;
    Switch* thinking_switch = nullptr;
    Slider* temp_slider = nullptr;
    Choice* ctx_choice = nullptr;
    Slider* tokens_slider = nullptr;
    Slider* gpu_slider = nullptr;
    Fl_Box* vram_note = nullptr;
    ActionBtn* close_btn = nullptr;

    ChatInspector(int X, int Y, int W, int H) : Fl_Group(X, Y, W, H) {
        box(FL_NO_BOX);

        close_btn = new ActionBtn(X + W - 38, Y + 12, 24, 24, "✕", ActionBtn::Secondary);
        close_btn->labelsize(11);

        preset_choice = new Choice(X + 16, Y + 68, W - 32, 28);
        preset_choice->add("Helpful Assistant (Default)|Coding Specialist|Concise & Direct|Creative Writer|Custom");
        preset_choice->value(0);

        sys_prompt = new Fl_Multiline_Input(X + 16, Y + 102, W - 32, 80);
        sys_prompt->box(theme::rounded);
        sys_prompt->color(fl_rgb_color(16, 20, 28));
        sys_prompt->textcolor(theme::text);
        sys_prompt->textsize(12);
        sys_prompt->wrap(1);

        apply_sys_btn = new ActionBtn(X + 16, Y + 188, W - 32, 26, "Apply System Prompt", ActionBtn::Primary);
        apply_sys_btn->labelsize(11);

        thinking_switch = new Switch(X + 16, Y + 252, W - 32, 26, "Deep Thinking / Reasoning");
        thinking_switch->labelsize(12);

        temp_slider = new Slider(X + 16, Y + 302, W - 32, 22);
        temp_slider->bounds(0.0, 2.0); temp_slider->step(0.05); temp_slider->value(0.70);

        ctx_choice = new Choice(X + 16, Y + 350, W - 32, 26);
        ctx_choice->add("2048|4096|8192|16384|32768");
        ctx_choice->value(1);

        tokens_slider = new Slider(X + 16, Y + 402, W - 32, 22);
        tokens_slider->bounds(64, 4096); tokens_slider->step(64); tokens_slider->value(512);

        gpu_slider = new Slider(X + 16, Y + 450, W - 32, 22);
        gpu_slider->bounds(0, 99); gpu_slider->step(1); gpu_slider->value(3);

        vram_note = new Fl_Box(X + 16, Y + 480, W - 32, 34, "Est. VRAM: 3.1 / 8.0 GiB (Partial offload)");
        vram_note->align(FL_ALIGN_LEFT | FL_ALIGN_INSIDE | FL_ALIGN_WRAP);
        vram_note->labelsize(11);
        vram_note->labelcolor(theme::accent);

        end();
    }

    void resize(int X, int Y, int W, int H) override {
        Fl_Group::resize(X, Y, W, H);
        if(close_btn) close_btn->position(X + W - 38, Y + 12);
        if(preset_choice) preset_choice->resize(X + 16, Y + 68, W - 32, 28);
        if(sys_prompt) sys_prompt->resize(X + 16, Y + 102, W - 32, 80);
        if(apply_sys_btn) apply_sys_btn->resize(X + 16, Y + 188, W - 32, 26);
        if(thinking_switch) thinking_switch->resize(X + 16, Y + 252, W - 32, 26);
        if(temp_slider) temp_slider->resize(X + 16, Y + 302, W - 32, 22);
        if(ctx_choice) ctx_choice->resize(X + 16, Y + 350, W - 32, 26);
        if(tokens_slider) tokens_slider->resize(X + 16, Y + 402, W - 32, 22);
        if(gpu_slider) gpu_slider->resize(X + 16, Y + 450, W - 32, 22);
        if(vram_note) vram_note->resize(X + 16, Y + 480, W - 32, 34);
    }

    void draw() override {
        fl_color(fl_rgb_color(14, 18, 26));
        fl_rectf(x(), y(), w(), h());
        fl_color(fl_rgb_color(26, 34, 48));
        fl_line(x(), y(), x(), y() + h());
        fl_line(x(), y() + 40, x() + w(), y() + 40);
        fl_line(x() + 16, y() + 224, x() + w() - 16, y() + 224);

        fl_color(theme::muted);
        fl_font(FL_HELVETICA_BOLD, 11);
        fl_draw("MODEL PARAMETERS", x() + 16, y() + 12, w() - 60, 24, FL_ALIGN_LEFT);
        fl_draw("SYSTEM PROMPT", x() + 16, y() + 46, w() - 32, 18, FL_ALIGN_LEFT);
        fl_draw("SAMPLING & MEMORY", x() + 16, y() + 230, w() - 32, 18, FL_ALIGN_LEFT);

        fl_font(FL_HELVETICA, 11);
        fl_draw("Temperature", x() + 16, y() + 284, w() - 32, 16, FL_ALIGN_LEFT);
        fl_draw("Context Length", x() + 16, y() + 332, w() - 32, 16, FL_ALIGN_LEFT);
        fl_draw("Max Tokens", x() + 16, y() + 384, w() - 32, 16, FL_ALIGN_LEFT);
        fl_draw("GPU Offload Layers", x() + 16, y() + 432, w() - 32, 16, FL_ALIGN_LEFT);

        Fl_Group::draw();
    }
};

class App {
public:
    Fl_Double_Window window{1200, 800, "FreeP100"};
    Fl_Group* tabs = nullptr;
    Fl_Group* pages[4]{};
    NavButton* navigation[4]{};
    Fl_Box* page_title = nullptr;
    Fl_Box* page_subtitle = nullptr;
    Fl_Box* top_server_chip = nullptr;
    ActionBtn* top_params_btn = nullptr;
    Fl_Box* status = nullptr;
    Fl_Box* bottom_ctx_chip = nullptr;

    // Header Model Capsule
    ModelCapsule* capsule = nullptr;
    Fl_Menu_Button* capsule_menu = nullptr;

    // Chat components
    Fl_Text_Buffer transcript, style_buf;
    ChatDisplay* chat_stream = nullptr;
    ChatDisplay* chat_view = nullptr; // pointer alias for compatibility
    ComposerCard* composer_card = nullptr;
    ChatInspector* chat_inspector = nullptr;
    bool chat_inspector_visible = false;
    LoadingCard* loading_card = nullptr;

    // Direct handles matching integration tests
    ChatInput* prompt = nullptr;
    ActionBtn *send = nullptr, *cancel = nullptr, *fresh = nullptr;
    ActionBtn *load = nullptr, *unload = nullptr;
    Choice* model_picker = nullptr;

    // Models tab components
    Fl_Hold_Browser* model_list = nullptr;
    Fl_Box* library_info = nullptr;
    Fl_Box* model_details_title = nullptr;
    Fl_Box* model_details_meta = nullptr;
    Fl_Box* model_fit_badge = nullptr;
    ActionBtn *details_load_btn = nullptr, *details_unload_btn = nullptr;

    // Tune tab components (Settings + Live Memory Plan side by side)
    Fl_Scroll* settings_scroll = nullptr;
    struct Section { Fl_Button* header; Fl_Group* body; int height; bool open; };
    std::vector<Section> sections;
    Fl_Box *profile_info = nullptr, *hardware_info = nullptr;
    Fl_Group* plan_panel = nullptr;
    Fl_Box *plan_model_size = nullptr, *plan_gpu_weights = nullptr, *plan_kv_cache = nullptr;
    Fl_Box *plan_vram_total = nullptr, *plan_ram_weights = nullptr, *plan_guard_notice = nullptr;
    Fl_Progress* plan_vram_gauge = nullptr;
    Fl_Text_Buffer review;
    Fl_Text_Display* review_view = nullptr;

    // Monitor tab components (Console & Live Logs)
    Fl_Box *hero_model = nullptr, *hero_tps = nullptr, *hero_ttft = nullptr, *hero_tokens = nullptr;
    Fl_Box *metric_vram = nullptr, *metric_ram = nullptr, *metric_context = nullptr, *metric_session = nullptr;
    Fl_Text_Buffer logs;
    Fl_Text_Display* log_view = nullptr;

    // Common fields and hardware state
    std::map<std::string, Fl_Input*> fields;
    std::map<std::string, Fl_Check_Button*> toggles;
    std::map<std::string, Fl_Choice*> choices;
    std::map<std::string, std::vector<std::string>> choice_values;
    std::map<std::string, Fl_Value_Slider*> sliders;

    ft::Hardware hardware;
    std::vector<ft::ModelInfo> library;
    std::vector<ft::ModelInfo> scanned;
    std::thread scanner;
    std::atomic<bool> scanning{false};
    bool needs_reload = false;
    int visiblePage = 0;

    // Telemetry and background poll state
    std::thread telem;
    std::mutex tlMutex;
    uint64_t vramTotal = 0, vramFree = 0, ramTotal = 0, ramFree = 0;
    bool vramKnown = false;
    int curSlots = 0, maxSlots = 0;
    std::atomic<double> lastTokSec{0.0};
    std::atomic<long long> lastTtftMs{0};
    uint64_t tokensIn = 0, tokensOut = 0, requestsTotal = 0;
    std::deque<std::string> logRing;
    std::atomic<bool> tlDirty{false}, logsDirty{false};
    std::string loadStage;
    int loadPercent = 0;
    std::atomic<bool> loadDirty{false};
    bool chatBusy = false;
    std::string lastStatsLine;
    std::chrono::steady_clock::time_point chatStarted;
    std::chrono::steady_clock::time_point serverStarted;

    Fl_Progress *vramGauge = nullptr, *ramGauge = nullptr;
    Fl_Box *vramGaugeLabel = nullptr, *ramGaugeLabel = nullptr;
    Fl_Box* chipLine = nullptr;

    ft::json config = ft::defaults(), messages = ft::json::array();
    std::filesystem::path data = ft::data_directory();
    ft::Process process;
    std::thread worker;
    std::atomic<bool> cancelled{false};
    std::mutex mutex;
    std::shared_ptr<httplib::Client> active_client;
    struct Update { std::string kind, text; };
    std::deque<Update> queue;
    bool busy = false, ready = false;
    std::atomic<bool> closing{false};
    int active_port = 0;
    std::string answer;
    bool integration = false;
    bool real_test = false;
    int integration_stage = 0;
    std::chrono::steady_clock::time_point integration_deadline;

    std::string active_model_display_name() {
        std::string p = config.value("model", std::string());
        if(p.empty()) return "No model loaded";
        auto name = std::filesystem::u8path(p).stem().u8string();
        return name;
    }

    void set_inspector(bool visible) {
        chat_inspector_visible = visible;
        if(top_params_btn) {
            top_params_btn->btn_style = chat_inspector_visible ? ActionBtn::Primary : ActionBtn::Secondary;
            top_params_btn->redraw();
        }
        layout_sections();
        window.redraw();
    }

    void toggle_inspector() {
        set_inspector(!chat_inspector_visible);
    }

    void navigate(int index) {
        if(index < 0 || index >= 4) return;
        for(int i = 0; i < 4; ++i) {
            if(pages[i]) {
                if(i == index) pages[i]->show();
                else pages[i]->hide();
            }
            if(navigation[i]) {
                navigation[i]->active = (i == index);
                navigation[i]->redraw();
            }
        }
        visiblePage = index;
        const char* titles[] = {"Chat", "Model Vault", "Tuning Lab", "Telemetry"};
        const char* subtitles[] = {"LOCAL INFERENCE SESSION", "GGUF LIBRARY & HARDWARE FIT", "PLACEMENT, MEMORY & SAMPLING", "LIVE ENGINE HEALTH & LOGS"};
        if(page_title) page_title->copy_label(titles[index]);
        if(page_subtitle) page_subtitle->copy_label(subtitles[index]);
        if(top_params_btn) {
            if(index == 0) top_params_btn->show();
            else top_params_btn->hide();
        }
        if(index == 2) update_memory_plan();
        window.redraw();
    }

    void on_server_line(const std::string& line) {
        {
            std::lock_guard<std::mutex> g(tlMutex);
            logRing.push_back(line);
            if(logRing.size() > 800) logRing.pop_front();
            logsDirty = true;
        }
        const char* stage = nullptr;
        int pct = -1;
        if(line.find("llama_model_loader:") != std::string::npos) { stage = "Reading model file"; pct = 5; }
        if(line.find("load_tensors: loading model tensors") != std::string::npos) { stage = "Loading tensors"; pct = 15; }
        if(line.find("load_tensors: offloaded") != std::string::npos) { stage = "Placing layers on GPU"; pct = 35; }
        if(line.find("llama_context:") != std::string::npos || line.find("warmup") != std::string::npos) { stage = "Allocating context and warmup"; pct = 70; }
        if(line.find("server is listening") != std::string::npos || line.find("listening on") != std::string::npos) { stage = "Starting API"; pct = 95; }
        if(stage) {
            std::lock_guard<std::mutex> g(tlMutex);
            loadStage = stage; loadPercent = pct; loadDirty = true;
        }
    }

    void telemetry_loop() {
        std::shared_ptr<httplib::Client> slotsClient;
        int cachedPort = 0;
        while (!closing) {
            auto gpu = ft::gpu_memory();
            auto ram = ft::ram_memory();
            {
                std::lock_guard<std::mutex> g(tlMutex);
                if (gpu.known) { vramTotal = gpu.total; vramFree = gpu.freeB; vramKnown = true; }
                ramTotal = ram.total; ramFree = ram.avail;
            }
            if (ready && active_port) {
                if (!slotsClient || cachedPort != active_port) {
                    slotsClient = std::make_shared<httplib::Client>("127.0.0.1", active_port);
                    slotsClient->set_connection_timeout(0, 300000);
                    slotsClient->set_read_timeout(0, 300000);
                    cachedPort = active_port;
                }
                auto res = slotsClient->Get("/slots");
                if (res && res->status == 200) {
                    try {
                        auto slots = ft::json::parse(res->body);
                        if (slots.is_array() && !slots.empty()) {
                            int busyCount = 0;
                            for (auto& s : slots) {
                                if (s.value("is_processing", false) || s.value("state", 0) != 0) busyCount++;
                            }
                            std::lock_guard<std::mutex> g(tlMutex);
                            curSlots = busyCount;
                            maxSlots = static_cast<int>(slots.size());
                        }
                    } catch(...) {}
                }
            }
            tlDirty = true;
            for(int i = 0; i < 10 && !closing; ++i) {
                std::this_thread::sleep_for(std::chrono::milliseconds(50));
            }
        }
    }

    void apply_telemetry() {
        uint64_t vT, vF, rT, rF;
        bool vK;
        int cS, mS;
        {
            std::lock_guard<std::mutex> g(tlMutex);
            vT = vramTotal; vF = vramFree; rT = ramTotal; rF = ramFree;
            vK = vramKnown; cS = curSlots; mS = maxSlots;
        }
        if(vramGauge && vramGaugeLabel) {
            if(vK && vT > 0) {
                uint64_t used = vT > vF ? vT - vF : 0;
                int pct = static_cast<int>(used * 100 / vT);
                vramGauge->value(std::clamp(pct, 0, 100));
                char b[64];
                snprintf(b, sizeof b, "VRAM %llu / %llu MiB (%d%%)", used / 1024 / 1024, vT / 1024 / 1024, pct);
                vramGaugeLabel->copy_label(b);
            } else {
                vramGaugeLabel->copy_label("VRAM (CPU Baseline)");
                vramGauge->value(0);
            }
        }
        if(ramGauge && ramGaugeLabel && rT > 0) {
            uint64_t used = rT > rF ? rT - rF : 0;
            int pct = static_cast<int>(used * 100 / rT);
            ramGauge->value(std::clamp(pct, 0, 100));
            char b[64];
            snprintf(b, sizeof b, "RAM %llu / %llu MiB (%d%%)", used / 1024 / 1024, rT / 1024 / 1024, pct);
            ramGaugeLabel->copy_label(b);
        }
        if(chipLine) {
            std::string ctxVal = config.value("ctx", "4096");
            std::ostringstream ss;
            ss << "CTX  " << ctxVal << "\n" << requestsTotal << " requests  ·  " << tokensOut << " out";
            chipLine->copy_label(ss.str().c_str());
        }
        if(bottom_ctx_chip) {
            std::string ctxVal = config.value("ctx", "4096");
            std::ostringstream ss;
            ss << "ctx " << ctxVal << " · " << requestsTotal << " reqs · " << tokensOut << " toks generated";
            bottom_ctx_chip->copy_label(ss.str().c_str());
        }
        if(composer_card && composer_card->ctx_chip) {
            std::string ctxVal = config.value("ctx", "4096");
            std::ostringstream ss;
            ss << "ctx " << ctxVal << " tokens";
            composer_card->ctx_chip->copy_label(ss.str().c_str());
        }
        auto fmt_mem = [](uint64_t used, uint64_t total) {
            if(total == 0) return std::string("Not available");
            std::ostringstream ss;
            ss << std::fixed << std::setprecision(1)
               << (double)used / (1024.0 * 1024.0 * 1024.0) << " / "
               << (double)total / (1024.0 * 1024.0 * 1024.0) << " GiB";
            return ss.str();
        };
        if(metric_vram) metric_vram->copy_label(vK ? fmt_mem(vT > vF ? vT - vF : 0, vT).c_str() : "CPU mode");
        if(metric_ram) metric_ram->copy_label(fmt_mem(rT > rF ? rT - rF : 0, rT).c_str());
        if(metric_context) {
            std::ostringstream ss;
            ss << config.value("ctx", "4096") << " tokens · " << cS << "/" << mS << " slots busy";
            metric_context->copy_label(ss.str().c_str());
        }
        if(metric_session) {
            std::ostringstream ss;
            ss << requestsTotal << " req · " << tokensIn << " in · " << tokensOut << " out";
            metric_session->copy_label(ss.str().c_str());
        }
        if(hero_ttft) {
            std::ostringstream ss;
            if(lastTtftMs.load() > 0) ss << lastTtftMs.load() << " ms · " << requestsTotal << " req";
            else ss << "— ms · " << requestsTotal << " req";
            hero_ttft->copy_label(ss.str().c_str());
        }
        if(hero_tokens) {
            std::ostringstream ss; ss << tokensIn << " in / " << tokensOut << " out";
            hero_tokens->copy_label(ss.str().c_str());
        }
    }

    void flush_logs() {
        std::vector<std::string> lines;
        {
            std::lock_guard<std::mutex> g(tlMutex);
            lines.assign(logRing.begin(), logRing.end());
        }
        std::string full;
        for(auto& l : lines) { full += l; full += "\n"; }
        logs.text(full.c_str());
        if(log_view) {
            log_view->insert_position(logs.length());
            log_view->show_insert_position();
        }
    }

    void update_capsule() {
        if(!capsule) return;
        std::string mname = active_model_display_name();
        std::string fit = "";
        try {
            std::string p = config.value("model", std::string());
            if(!p.empty()) {
                auto m = ft::inspect_model(std::filesystem::u8path(p));
                long long ngl = 0; try { ngl = std::stoll(config.value("gpu_layers", "0")); } catch(...) {}
                long long layers = std::max<long long>(1, (long long)m.layers);
                long long gpuLayers = std::clamp(ngl >= 999 ? layers : ngl, 0LL, layers);
                fit = std::to_string(gpuLayers) + "/" + std::to_string(layers) + " GPU";
            }
        } catch(...) {}
        capsule->update(p_model_loaded() ? mname : "", fit, ready, busy, loadPercent);
    }

    bool p_model_loaded() {
        return !config.value("model", std::string()).empty();
    }

    void set_status(const std::string& message) {
        if(status) status->copy_label(message.c_str());
        update_capsule();
    }

    void post(std::string kind, std::string text = "") {
        std::lock_guard<std::mutex> guard(mutex);
        queue.push_back({std::move(kind), std::move(text)});
        Fl::awake();
    }

    static void poll_cb(void* v) {
        auto* a = static_cast<App*>(v);
        a->poll();
        if(!a->closing) Fl::repeat_timeout(0.05, poll_cb, v);
    }
    static void load_cb(Fl_Widget*, void* v) { static_cast<App*>(v)->start(); }
    static void edited_cb(Fl_Widget*, void* v) {
        auto* a = static_cast<App*>(v);
        a->needs_reload = a->ready;
        a->profile_state();
        a->update_memory_plan();
        a->save();
    }

    ft::json values() {
        auto c = config;
        for(auto& f : fields) c[f.first] = f.second->value();
        for(auto& f : toggles) c[f.first] = f.second->value() ? "1" : "0";
        for(auto& f : choices) {
            int v = f.second->value();
            auto& opts = choice_values.at(f.first);
            if(v >= 0 && v < static_cast<int>(opts.size())) c[f.first] = opts[v];
        }
        for(auto& f : sliders) {
            std::ostringstream s;
            if(f.first == "gpu_layers" && f.second->value() >= f.second->maximum()) s << "999";
            else s << std::fixed << std::setprecision(f.first == "temperature" ? 2 : 0) << f.second->value();
            c[f.first] = s.str();
        }
        return c;
    }

    void profile_state() {
        if(!profile_info) return;
        auto rec = config["recommended"];
        if(rec.empty()) {
            profile_info->copy_label("RECOMMENDED  /  Hardware baseline");
            profile_info->labelcolor(theme::muted);
            return;
        }
        bool dirty = false;
        for(auto it = rec.begin(); it != rec.end(); ++it) {
            if(fields.count(it.key()) && fields[it.key()]->value() != it.value().get<std::string>()) dirty = true;
            if(toggles.count(it.key()) && (toggles[it.key()]->value() ? "1" : "0") != it.value().get<std::string>()) dirty = true;
            if(sliders.count(it.key())) {
                auto v = it.value().get<std::string>();
                if(it.key() == "gpu_layers" && v == "999") {
                    if(sliders[it.key()]->value() != sliders[it.key()]->maximum()) dirty = true;
                } else if(std::abs(sliders[it.key()]->value() - std::stod(v)) > 0.001) dirty = true;
            }
        }
        if(dirty) {
            profile_info->copy_label("CUSTOMIZED  /  Settings differ from recommendation");
            profile_info->labelcolor(theme::status_amber);
        } else {
            profile_info->copy_label("RECOMMENDED  /  Matches hardware starting point");
            profile_info->labelcolor(theme::accent);
        }
    }

    void restore_profile() {
        if(config["recommended"].empty()) return;
        for(auto it = config["recommended"].begin(); it != config["recommended"].end(); ++it) {
            if(fields.count(it.key())) fields[it.key()]->value(it.value().get<std::string>().c_str());
            if(toggles.count(it.key())) toggles[it.key()]->value(it.value().get<std::string>() == "1");
            if(sliders.count(it.key())) {
                auto v = it.value().get<std::string>();
                if(it.key() == "gpu_layers" && v == "999") sliders[it.key()]->value(sliders[it.key()]->maximum());
                else sliders[it.key()]->value(std::stod(v));
            }
        }
        needs_reload = ready;
        profile_state();
        update_memory_plan();
        save();
        set_status("Recommended settings restored. Load / reload to apply.");
    }

    void update_memory_plan() {
        review.text(plan_summary().c_str());
        update_memory_plan_ui();
    }

    void update_memory_plan_ui() {
        try {
            auto modelPath = std::filesystem::u8path(config.value("model", std::string()));
            if(!std::filesystem::is_regular_file(modelPath)) return;
            auto m = ft::inspect_model(modelPath);
            long long ngl = 0; try { ngl = std::stoll(config.value("gpu_layers", "0")); } catch(...) {}
            long long layers = std::max<long long>(1, (long long)m.layers);
            long long gpuLayers = std::clamp(ngl >= 999 ? layers : ngl, 0LL, layers);
            double frac = (double)gpuLayers / layers;
            auto val = [&](const char* k) -> uint64_t { try { return (uint64_t)std::stoull(config.value(k, "0")); } catch(...) { return 0; } };
            double headDim = m.heads ? (double)m.embedding / (double)m.heads : 128.0;
            double perTok = (double)m.layers * (m.kv_heads ? m.kv_heads : m.heads) * headDim * 4.0;
            long long ctx = 0; try { ctx = std::stoll(config.value("ctx", "4096")); } catch(...) {}
            uint64_t kvBytes = (uint64_t)(perTok * (double)ctx);
            uint64_t wG = (uint64_t)((double)m.bytes * frac), wC = m.bytes - wG;

            if(plan_model_size) {
                std::ostringstream ss;
                ss << "Model: " << std::fixed << std::setprecision(1) << (double)m.bytes / (1024*1024*1024)
                   << " GiB (" << gpuLayers << "/" << layers << " GPU layers)";
                plan_model_size->copy_label(ss.str().c_str());
            }
            if(plan_gpu_weights) {
                std::ostringstream ss;
                ss << "GPU Weights: " << std::fixed << std::setprecision(1) << (double)wG / (1024*1024*1024) << " GiB";
                plan_gpu_weights->copy_label(ss.str().c_str());
            }
            if(plan_kv_cache) {
                std::ostringstream ss;
                ss << "KV Cache: " << std::fixed << std::setprecision(1) << (double)kvBytes / (1024*1024*1024)
                   << " GiB (" << ctx << " ctx)";
                plan_kv_cache->copy_label(ss.str().c_str());
            }
            if(plan_ram_weights) {
                std::ostringstream ss;
                ss << "RAM Weights: " << std::fixed << std::setprecision(1) << (double)wC / (1024*1024*1024) << " GiB";
                plan_ram_weights->copy_label(ss.str().c_str());
            }

            uint64_t estVram = wG + kvBytes + (val("gpu_cache") * 1024 * 1024) + (val("reserve") * 1024 * 1024);
            uint64_t totalVram = hardware.vram ? hardware.vram : (8ull * 1024 * 1024 * 1024);
            int pct = static_cast<int>(estVram * 100 / totalVram);
            if(plan_vram_gauge) plan_vram_gauge->value(std::clamp(pct, 0, 100));
            if(plan_vram_total) {
                std::ostringstream ss;
                ss << "Estimated VRAM: " << std::fixed << std::setprecision(1)
                   << (double)estVram / (1024*1024*1024) << " GiB / "
                   << (double)totalVram / (1024*1024*1024) << " GiB";
                plan_vram_total->copy_label(ss.str().c_str());
            }
            if(chat_inspector && chat_inspector->vram_note) {
                std::ostringstream ss;
                ss << "Est. VRAM: " << std::fixed << std::setprecision(1) << (double)estVram / (1024*1024*1024)
                   << " / " << (double)totalVram / (1024*1024*1024) << " GiB (" << pct << "% offload)";
                chat_inspector->vram_note->copy_label(ss.str().c_str());
            }
        } catch(...) {}
    }

    void scan_folder() {
        if(scanning) return;
        std::filesystem::path folder = std::filesystem::u8path(fields.at("model_folder")->value());
        if(!std::filesystem::exists(folder)) { set_status("Folder does not exist."); return; }
        scanning = true; set_status("Scanning for models...");
        if(scanner.joinable()) scanner.join();
        scanner = std::thread([this, folder] {
            try {
                auto list = ft::scan_models(folder);
                { std::lock_guard<std::mutex> lock(mutex); scanned = std::move(list); }
                post("library", "Models refreshed.");
            } catch(const std::exception& e) { post("library_error", e.what()); }
        });
    }

    void select_model(int index) {
        if(index < 0 || index >= static_cast<int>(library.size())) return;
        auto& m = library[index];
        fields.at("model")->value(m.path.u8string().c_str());
        fields.at("mmproj")->value(m.projector.u8string().c_str());
        config["model"] = m.path.u8string();
        config["mmproj"] = m.projector.u8string();

        if(model_list) model_list->value(index + 1);

        if(model_details_title) model_details_title->copy_label(m.path.stem().u8string().c_str());
        if(model_details_meta) {
            std::ostringstream s;
            s << "Architecture: " << (m.architecture.empty() ? "unknown" : m.architecture)
              << "\nFile size: " << m.bytes / 1024 / 1024 << " MiB (" << std::fixed << std::setprecision(1) << (double)m.bytes / (1024*1024*1024) << " GiB)"
              << "\nLayers: " << m.layers << "   ·   Max Context: " << m.context
              << "\nPath: " << m.path.u8string()
              << "\nVision Projector: " << (m.projector.empty() ? "None paired" : m.projector.filename().u8string());
            model_details_meta->copy_label(s.str().c_str());
        }
        if(model_fit_badge) {
            std::ostringstream s;
            s << "PROFILE READY  ·  " << m.layers << " layers  ·  "
              << (m.context >= 1024 ? std::to_string(m.context / 1024) + "k" : std::to_string(m.context))
              << " max context  ·  review exact placement in Tuning Lab";
            model_fit_badge->copy_label(s.str().c_str());
        }

        if(toggles.at("recommend")->value()) {
            auto rec = ft::recommend(m, hardware);
            for(auto it = rec.begin(); it != rec.end(); ++it) {
                if(fields.count(it.key())) fields[it.key()]->value(it.value().get<std::string>().c_str());
                if(toggles.count(it.key())) toggles[it.key()]->value(it.value().get<std::string>() == "1");
                if(sliders.count(it.key())) {
                    auto v = it.value().get<std::string>();
                    if(it.key() == "gpu_layers" && v == "999") sliders[it.key()]->value(sliders[it.key()]->maximum());
                    else sliders[it.key()]->value(std::stod(v));
                }
            }
            config["recommended"] = rec;
            config["recommended_model"] = m.path.u8string();
        }

        needs_reload = ready;
        profile_state();
        update_memory_plan();
        save();
        update_capsule();
        set_status("Model selected: " + m.path.stem().u8string() + ". Ready to load.");
    }

    void populate_library() {
        if(model_picker) model_picker->clear();
        if(model_list) model_list->clear();
        if(capsule_menu) capsule_menu->clear();

        for(size_t i = 0; i < library.size(); ++i) {
            auto& m = library[i];
            std::string label = m.path.stem().u8string();
            std::string sizeStr = " (" + std::to_string(m.bytes / 1024 / 1024) + " MiB)";
            if(model_picker) model_picker->add((label + sizeStr).c_str());
            if(model_list) model_list->add((label + "  /  " + std::to_string(m.bytes / 1024 / 1024) + " MiB").c_str());
            if(capsule_menu) {
                capsule_menu->add(label.c_str(), 0, [](Fl_Widget*, void* p) {
                    auto* pair = static_cast<std::pair<App*, int>*>(p);
                    pair->first->select_model(pair->second);
                    delete pair;
                }, new std::pair<App*, int>(this, static_cast<int>(i)));
            }
        }
        if(library_info) {
            uint64_t total = 0;
            size_t vision = 0, moe = 0;
            for(const auto& m : library) {
                total += m.bytes;
                if(!m.projector.empty()) vision++;
                if(m.experts > 0) moe++;
            }
            std::ostringstream ss;
            ss << library.size() << " models  ·  " << std::fixed << std::setprecision(1)
               << (double)total / (1024.0 * 1024.0 * 1024.0) << " GiB indexed  ·  "
               << moe << " MoE  ·  " << vision << " vision-ready";
            library_info->copy_label(ss.str().c_str());
        }
        if(navigation[1]) {
            static std::string countBadge;
            countBadge = std::to_string(library.size());
            navigation[1]->badge = countBadge.c_str();
            navigation[1]->redraw();
        }
        std::string selected = config.value("model", std::string());
        for(size_t i = 0; i < library.size(); ++i) {
            if(library[i].path.u8string() == selected) {
                if(model_picker) model_picker->value(static_cast<int>(i));
                if(model_list) model_list->value(static_cast<int>(i + 1));
                select_model(static_cast<int>(i));
                break;
            }
        }
        update_capsule();
    }

    void layout_sections() {
        int W = window.w(), H = window.h();
        int top_h = 50;
        int sidebar_w = 200;
        int client_w = W - sidebar_w;
        int client_h = H - top_h - 28;

        if(page_title) page_title->resize(sidebar_w + 16, 6, 180, 26);
        if(page_subtitle) page_subtitle->resize(sidebar_w + 16, 31, 220, 13);
        if(capsule) {
            int cap_w = std::min(520, client_w - 240);
            int cap_x = sidebar_w + (client_w - cap_w) / 2;
            capsule->resize(cap_x, 8, cap_w, 36);
            if(capsule_menu) capsule_menu->resize(cap_x, 8, cap_w - 232, 36);
        }
        if(top_params_btn) top_params_btn->resize(W - 130, 10, 114, 30);
        if(top_server_chip) top_server_chip->resize(W - 250, 10, 110, 30);

        if(tabs) tabs->resize(sidebar_w, top_h, client_w, client_h);
        for(int i = 0; i < 4; ++i) if(pages[i]) pages[i]->resize(sidebar_w, top_h, client_w, client_h);

        // Chat View Layout
        int insp_w = chat_inspector_visible ? 280 : 0;
        int stream_w = client_w - insp_w;

        if(chat_stream) chat_stream->resize(sidebar_w + 12, top_h + 8, stream_w - 24, client_h - 124);
        if(composer_card) composer_card->resize(sidebar_w + 16, top_h + client_h - 110, stream_w - 32, 100);
        if(chat_inspector) {
            if(chat_inspector_visible) {
                chat_inspector->resize(sidebar_w + stream_w, top_h, insp_w, client_h);
                chat_inspector->show();
            } else {
                chat_inspector->hide();
            }
        }
        if(loading_card) {
            int lw = std::min(600, stream_w - 40);
            loading_card->resize(sidebar_w + (stream_w - lw) / 2, top_h + 80, lw, 120);
        }

        // Tune View Layout
        int split_w = client_w / 2;
        if(settings_scroll) settings_scroll->resize(sidebar_w + 12, top_h + 88, split_w - 20, client_h - 108);
        if(plan_panel) plan_panel->resize(sidebar_w + split_w + 6, top_h + 88, split_w - 18, client_h - 108);

        // Models View Layout
        int list_w = std::min(460, client_w / 2);
        if(model_list) model_list->resize(sidebar_w + 16, top_h + 92, list_w - 24, client_h - 162);

        // Monitor View Layout
        if(log_view) log_view->resize(sidebar_w + 16, top_h + 176, client_w - 32, client_h - 254);
    }

    void set_busy(bool value) {
        busy = value;
        update_capsule();
        if(value) {
            if(send) send->deactivate();
            if(fresh) fresh->deactivate();
            if(cancel) { cancel->show(); cancel->activate(); }
            if(load) load->deactivate();
            if(unload) unload->deactivate();
            if(details_load_btn) details_load_btn->deactivate();
            if(details_unload_btn) details_unload_btn->deactivate();
        } else {
            if(cancel) cancel->hide();
            if(load) load->activate();
            if(fresh) fresh->activate();
            if(details_load_btn) details_load_btn->activate();
            if(ready) {
                if(send) send->activate();
                if(unload) unload->activate();
                if(details_unload_btn) details_unload_btn->activate();
            } else {
                if(send) send->deactivate();
                if(unload) unload->deactivate();
                if(details_unload_btn) details_unload_btn->deactivate();
            }
        }
        if(needs_reload && send) send->deactivate();
    }

    ft::json collect() {
        auto c = values();
        ft::validate(c);
        return c;
    }

    void show_config() {
        for(auto& f : fields) if(config.contains(f.first) && config[f.first].is_string()) f.second->value(config[f.first].get<std::string>().c_str());
        for(auto& f : toggles) f.second->value(config.value(f.first, "0") == "1");
        for(auto& f : choices) {
            auto& opts = choice_values.at(f.first);
            auto v = config.value(f.first, std::string());
            auto it = std::find(opts.begin(), opts.end(), v);
            if(it == opts.end()) {
                f.second->add(v.c_str()); opts.push_back(v);
                f.second->value(static_cast<int>(opts.size() - 1));
            } else f.second->value(static_cast<int>(it - opts.begin()));
        }
        for(auto& f : sliders) try {
            auto v = std::stod(config.at(f.first).get<std::string>());
            if(f.first == "gpu_layers" && v == 999) v = f.second->maximum();
            f.second->maximum(std::max(f.second->maximum(), v));
            f.second->value(v);
        } catch(...) {}
        if(hardware_info) {
            std::ostringstream s;
            s << hardware.gpu << "   /   " << hardware.vram / 1024 / 1024 << " MiB VRAM   /   "
              << hardware.threads << " CPU threads   /   " << hardware.ram / 1024 / 1024 << " MiB available RAM";
            hardware_info->copy_label(s.str().c_str());
        }
        if(chat_inspector && chat_inspector->sys_prompt && config.contains("system")) {
            chat_inspector->sys_prompt->value(config["system"].get<std::string>().c_str());
        }
        profile_state();
    }

    void save() {
        config = collect();
        ft::save_json(data / "settings.json", config);
    }
    void save_history() {
        ft::save_json(data / "conversation.json", messages);
    }

    void choose(const char* key, bool executable) {
        Fl_Native_File_Chooser chooser;
        chooser.title(executable ? "Select llama-server executable" : "Select GGUF model");
        chooser.type(Fl_Native_File_Chooser::BROWSE_FILE);
        if(!executable) chooser.filter("GGUF model\t*.gguf");
        if(chooser.show() == 0) {
            if(std::string(key) == "model") {
                auto m = ft::inspect_model(std::filesystem::u8path(chooser.filename()));
                if(!m.valid || m.is_projector) { set_status("Choose a valid language-model GGUF, not a projector."); return; }
                library.push_back(m); populate_library();
                select_model(static_cast<int>(library.size() - 1));
            } else {
                if(std::string(key) == "mmproj") {
                    auto m = ft::inspect_model(std::filesystem::u8path(chooser.filename()));
                    if(!m.valid || !m.is_projector) { set_status("Choose a readable vision projector GGUF."); return; }
                    for(auto& model : library) if(model.path.u8string() == values()["model"]) model.projector = m.path;
                    populate_library(); if(ready) needs_reload = true;
                }
                fields.at(key)->value(chooser.filename()); edited_cb(nullptr, this);
            }
        }
    }

    void stop_request() {
        cancelled = true;
        std::lock_guard<std::mutex> guard(mutex);
        if(active_client) active_client->stop();
    }

    void new_chat() {
        if(busy) return;
        if(!messages.empty() && fl_choice("Clear the saved conversation?", "Cancel", "Clear", nullptr) != 1) return;
        messages = ft::json::array();
        transcript.text("");
        style_buf.text("");
        if(chat_stream) chat_stream->clear_messages();
        try { save_history(); set_status("New conversation started"); }
        catch(const std::exception& e) { set_status(e.what()); }
    }

    void start() {
        if(busy) return;
        try {
            save();
            auto args = ft::arguments(config);
            auto c = config;
            args.push_back("--metrics");
            args.push_back("--slots");
            if(worker.joinable()) worker.join();
            cancelled = false; set_busy(true); ready = false;
            set_status("Loading model - Stop cancels startup");

            if(loading_card) {
                std::string mname = active_model_display_name();
                loading_card->title_box->copy_label(("Loading Model: " + mname).c_str());
                loading_card->stage_box->copy_label("Starting server process...");
                loading_card->progress_bar->value(5);
                loading_card->show();
                window.redraw();
            }

            worker = std::thread([this, args, c] {
                try {
                    process.stop();
                    int port = std::stoi(c.at("port").get<std::string>());
                    httplib::Server port_probe;
                    if(!port_probe.bind_to_port("127.0.0.1", port)) throw std::runtime_error("Port is already in use. Choose another port.");
                    port_probe.stop();
                    process.start(args, data / "server.log", c.at("graphs") == "off",
                        [this](const std::string& line) { on_server_line(line); });
                    auto client = std::make_shared<httplib::Client>("127.0.0.1", port);
                    client->set_connection_timeout(1, 0); client->set_read_timeout(1, 0);
                    { std::lock_guard<std::mutex> guard(mutex); active_client = client; }
                    bool healthy = false;
                    auto deadline = std::chrono::steady_clock::now() + std::chrono::minutes(5);
                    while(!cancelled && std::chrono::steady_clock::now() < deadline) {
                        if(!process.running()) throw std::runtime_error("Server exited during startup. See Diagnostics.");
                        auto response = client->Get("/health");
                        if(response && response->status == 200) { healthy = true; break; }
                        std::this_thread::sleep_for(std::chrono::milliseconds(200));
                    }
                    if(cancelled || !healthy) {
                        process.stop();
                        post("load_error", cancelled ? "Startup cancelled" : "Startup timed out. See Diagnostics.");
                    } else {
                        serverStarted = std::chrono::steady_clock::now();
                        post("ready", std::to_string(port));
                    }
                } catch(const std::exception& e) {
                    process.stop(); post("load_error", e.what());
                }
            });
        } catch(const std::exception& e) { set_status(e.what()); }
    }

    void chat() {
        if(busy || !ready || needs_reload || !prompt || std::string(prompt->value()).empty()) return;
        try {
            auto c = collect();
            std::string user = prompt->value();
            if(user.size() > 200000 || messages.dump().size() > 2 * 1024 * 1024) throw std::runtime_error("Conversation limit reached. Start a new conversation.");
            messages.push_back({{"role", "user"}, {"content", user}});
            auto request_messages = messages;
            if(c.at("system") != "") request_messages.insert(request_messages.begin(), ft::json{{"role", "system"}, {"content", c.at("system")}});

            ft::json body = {
                {"model", "local"}, {"messages", request_messages}, {"stream", true},
                {"temperature", std::stod(c.at("temperature").get<std::string>())},
                {"max_tokens", std::stoi(c.at("max_tokens").get<std::string>())},
                {"chat_template_kwargs", {{"enable_thinking", c.at("thinking") == "1"}}},
                {"top_k", std::stoi(c.at("top_k").get<std::string>())},
                {"top_p", std::stod(c.at("top_p").get<std::string>())},
                {"min_p", std::stod(c.at("min_p").get<std::string>())},
                {"repeat_penalty", std::stod(c.at("repeat_penalty").get<std::string>())},
                {"timings_per_token", true},
                {"stream_options", {{"include_usage", true}}}
            };

            // Add user card and start assistant card
            if(chat_stream) {
                chat_stream->add_user_message(user);
                chat_stream->start_assistant_message(active_model_display_name());
            }

            prompt->value(""); answer.clear();
            if(worker.joinable()) worker.join();
            cancelled = false; set_busy(true); set_status("Generating...");
            chatStarted = std::chrono::steady_clock::now(); chatBusy = true;
            lastTokSec = 0; lastTtftMs = 0;
            int port = active_port;

            worker = std::thread([this, body, port] {
                try {
                    auto client = std::make_shared<httplib::Client>("127.0.0.1", port);
                    client->set_connection_timeout(2, 0); client->set_read_timeout(300, 0);
                    { std::lock_guard<std::mutex> guard(mutex); active_client = client; }
                    size_t total = 0;
                    bool firstToken = true;
                    ft::json usage;

                    ft::Events events([&](const ft::json& event) {
                        if(event.contains("error")) throw std::runtime_error(event["error"].dump());
                        if(event.contains("usage") && event["usage"].is_object()) usage = event["usage"];
                        if(!event.contains("choices") || event["choices"].empty()) {
                            if(event.contains("timings")) {
                                auto& t = event["timings"];
                                lastTokSec = t.value("predicted_per_second", 0.0);
                            }
                            return;
                        }
                        auto delta = event["choices"][0].value("delta", ft::json::object());
                        if(firstToken) {
                            firstToken = false;
                            lastTtftMs = std::chrono::duration_cast<std::chrono::milliseconds>(
                                std::chrono::steady_clock::now() - chatStarted).count();
                        }
                        if(delta.contains("reasoning_content") && delta["reasoning_content"].is_string()) {
                            auto text = delta["reasoning_content"].get<std::string>(); total += text.size();
                            if(total > 1024 * 1024) throw std::runtime_error("Response display limit reached (1 MiB)");
                            post("reasoning", text);
                        }
                        if(delta.contains("content") && delta["content"].is_string()) {
                            auto token = delta["content"].get<std::string>(); total += token.size();
                            if(total > 1024 * 1024) throw std::runtime_error("Response display limit reached (1 MiB)");
                            post("token", token);
                        }
                    });

                    httplib::Request request; request.method = "POST"; request.path = "/v1/chat/completions";
                    request.body = body.dump(); request.set_header("Content-Type", "application/json");
                    int http_status = 0; std::string error_body;
                    request.response_handler = [&](const httplib::Response& r) { http_status = r.status; return true; };
                    request.content_receiver = [&](const char* bytes, size_t n, uint64_t, uint64_t) {
                        if(cancelled) return false;
                        if(http_status != 200) {
                            if(error_body.size() < 8192) error_body.append(bytes, std::min(n, size_t(8192 - error_body.size())));
                        } else events.feed(bytes, n);
                        return true;
                    };

                    auto response = client->send(request);
                    if(usage.is_object()) {
                        tokensIn += usage.value("prompt_tokens", 0);
                        tokensOut += usage.value("completion_tokens", 0);
                    }
                    requestsTotal++;
                    {
                        std::string stop = cancelled ? "stopped" : (http_status == 200 ? (events.done ? "completed" : "truncated") : "failed");
                        char buf[160];
                        if(usage.is_object()) {
                            double tps = lastTokSec.load();
                            long long ttft = lastTtftMs.load();
                            int toks = usage.value("completion_tokens", 0);
                            snprintf(buf, sizeof buf, "[ ⚡ %.1f tok/s  ·  ⏱ TTFT %lld ms  ·  📊 %d tokens  ·  %s ]", tps, ttft, toks, stop.c_str());
                        } else snprintf(buf, sizeof buf, "[ %s ]", stop.c_str());
                        lastStatsLine = buf;
                    }
                    if(cancelled) post("chat_done", "Stopped. Partial response saved.");
                    else if(!response) post("chat_done", "Request failed: " + httplib::to_string(response.error()));
                    else if(http_status != 200) post("chat_done", "HTTP " + std::to_string(http_status) + ": " + error_body);
                    else if(!events.done) post("chat_done", "Stream ended before completion. Partial response saved.");
                    else post("chat_done", "Ready");
                } catch(const std::exception& e) { post("chat_done", e.what()); }
            });
        } catch(const std::exception& e) { set_status(e.what()); }
    }

    void refresh_log() {
        std::ifstream in(data / "server.log", std::ios::binary);
        if(!in) { logs.text("No server log yet."); return; }
        in.seekg(0, std::ios::end); auto size = in.tellg();
        in.seekg(size > 262144 ? size - std::streamoff(262144) : std::streampos(0));
        std::string text((std::istreambuf_iterator<char>(in)), {});
        logs.text(text.c_str());
    }

    std::string plan_summary() {
        try {
            auto modelPath = std::filesystem::u8path(config.value("model", std::string()));
            if(!std::filesystem::is_regular_file(modelPath)) return "";
            auto m = ft::inspect_model(modelPath);
            long long ngl = 0; try { ngl = std::stoll(config.value("gpu_layers", "0")); } catch(...) {}
            long long layers = std::max<long long>(1, (long long)m.layers);
            long long gpuLayers = std::clamp(ngl >= 999 ? layers : ngl, 0LL, layers);
            double frac = (double)gpuLayers / layers;
            auto val = [&](const char* k) -> uint64_t { try { return (uint64_t)std::stoull(config.value(k, "0")); } catch(...) { return 0; } };
            double headDim = m.heads ? (double)m.embedding / (double)m.heads : 128.0;
            double perTok = (double)m.layers * (m.kv_heads ? m.kv_heads : m.heads) * headDim * 4.0;
            long long ctx = 0; try { ctx = std::stoll(config.value("ctx", "4096")); } catch(...) {}
            uint64_t kvBytes = (uint64_t)(perTok * (double)ctx);
            uint64_t wG = (uint64_t)((double)m.bytes * frac), wC = m.bytes - wG;
            bool expertOn = config.value("expert_enabled", "1") == "1";
            std::ostringstream s;
            s << "\nEstimated memory plan (not calibrated):\n";
            s << "  Model: " << m.bytes / 1024 / 1024 << " MiB, " << gpuLayers << " of " << layers << " layers on GPU\n";
            if(gpuLayers > 0) {
                s << "  GPU: weights ~" << wG / 1024 / 1024 << " MiB";
                if(expertOn && val("gpu_cache")) s << ", expert cache " << val("gpu_cache") << " MiB";
                if(kvBytes) s << ", KV ~" << kvBytes / 1024 / 1024 << " MiB";
                s << ", reserve " << val("reserve") << " MiB\n";
            }
            if(wC || val("cpu_cache") || (kvBytes && gpuLayers == 0)) {
                s << "  RAM:";
                if(wC) s << " weights ~" << wC / 1024 / 1024 << " MiB";
                if(expertOn && val("cpu_cache")) s << ", expert cache " << val("cpu_cache") << " MiB";
                if(kvBytes && gpuLayers == 0) s << ", KV ~" << kvBytes / 1024 / 1024 << " MiB";
                s << "\n";
            }
            s << "  Scratch, CUDA graphs and other workloads are extra.\n";
            return s.str();
        } catch(...) { return ""; }
    }

    void poll() {
        std::deque<Update> changes;
        { std::lock_guard<std::mutex> guard(mutex); changes.swap(queue); }

        if(loadDirty.exchange(false) && busy && !ready) {
            std::string stage; int pct = 0;
            { std::lock_guard<std::mutex> g(tlMutex); stage = loadStage; pct = loadPercent; }
            set_status("Loading - " + stage + " (" + std::to_string(std::max(0, pct)) + "%)");
            if(loading_card) {
                loading_card->stage_box->copy_label((stage + " (" + std::to_string(std::max(0, pct)) + "%)").c_str());
                loading_card->progress_bar->value(std::max(0, pct));
                loading_card->show();
                loading_card->redraw();
            }
            update_capsule();
        }

        if(chatBusy && lastTokSec > 0.0) {
            char b[64]; snprintf(b, sizeof b, "⚡ %.1f tok/s", lastTokSec.load());
            set_status(std::string("Generating... ") + b);
            if(composer_card && composer_card->speed_chip) {
                composer_card->speed_chip->copy_label(b);
                composer_card->speed_chip->show();
                composer_card->speed_chip->redraw();
            }
            if(hero_tps) {
                char tb[32]; snprintf(tb, sizeof tb, "%.1f tok/s", lastTokSec.load());
                hero_tps->copy_label(tb); hero_tps->redraw();
            }
        } else if(!chatBusy && composer_card && composer_card->speed_chip && composer_card->speed_chip->visible()) {
            composer_card->speed_chip->hide();
        }

        if(tlDirty.exchange(false)) apply_telemetry();
        if(logsDirty.exchange(false) && visiblePage == 3) flush_logs();

        for(auto& u : changes) {
            if(u.kind == "library" || u.kind == "library_error") {
                if(scanner.joinable()) scanner.join();
                scanning = false;
                if(u.kind == "library_error") set_status(u.text);
                else {
                    { std::lock_guard<std::mutex> lock(mutex); library = std::move(scanned); }
                    populate_library();
                    config["model_folder"] = fields.at("model_folder")->value();
                    try { save(); } catch(const std::exception& e) { set_status(e.what()); }
                    set_status(std::to_string(library.size()) + " models found. Select one to see details.");
                }
                continue;
            }
            if(u.kind == "token") {
                answer += u.text;
                if(chat_stream) chat_stream->append_token(u.text);
            } else if(u.kind == "reasoning") {
                if(chat_stream) chat_stream->append_reasoning(u.text);
                set_status("Thinking...");
            } else {
                if(worker.joinable()) worker.join();
                { std::lock_guard<std::mutex> guard(mutex); active_client.reset(); }
                if(u.kind == "ready") {
                    ready = true; needs_reload = false; active_port = std::stoi(u.text);
                    set_status("Ready - model loaded on localhost:" + u.text);
                    if(loading_card) loading_card->hide();
                    if(hero_model) hero_model->copy_label((active_model_display_name() + " (● Running)").c_str());
                    if(top_server_chip) top_server_chip->copy_label(("● :" + u.text).c_str());
                } else if(u.kind == "load_error") {
                    ready = false; set_status(u.text); refresh_log();
                    if(loading_card) loading_card->hide();
                    if(hero_model) hero_model->copy_label((active_model_display_name() + " (○ Stopped)").c_str());
                    if(top_server_chip) top_server_chip->copy_label("○ Server stopped");
                } else if(u.kind == "chat_done") {
                    chatBusy = false;
                    if(!answer.empty()) messages.push_back({{"role", "assistant"}, {"content", answer}});
                    if(chat_stream) chat_stream->finish_assistant_message(lastStatsLine);
                    lastStatsLine.clear();
                    set_status(u.text);
                    try { save_history(); } catch(const std::exception& e) { set_status(e.what()); }
                }
                set_busy(false);
            }
        }
        if(!busy && ready && !process.running()) {
            ready = false; set_busy(false); set_status("Server exited. See Diagnostics.");
            if(hero_model) hero_model->copy_label((active_model_display_name() + " (○ Stopped)").c_str());
        }

        // Integration test logic
        if(integration) {
            if(integration_stage == 0 && ready && !busy) {
                integration_stage = 1;
                prompt->value("Exercise streaming through the desktop client.");
                chat();
            }
            if(integration_stage == 1 && !busy && std::string(status->label()) == "Ready" && (real_test ? !answer.empty() : answer == "Native stream verified.")) {
                if(real_test) {
                    ft::save_json(data / "integration-result.json", {{"passed", true}, {"response", answer}});
                    integration_stage = 5; closing = true; window.hide();
                } else {
                    integration_stage = 2; start();
                }
            }
            if(integration_stage == 2 && ready && !busy && messages.size() == 2) {
                integration_stage = 3;
                prompt->value("Cancel this streaming request.");
                chat();
            }
            if(integration_stage == 3 && busy && !answer.empty()) {
                integration_stage = 4;
                stop_request();
            }
            if(integration_stage == 4 && !busy && messages.size() == 4 && std::string(status->label()).rfind("Stopped.", 0) == 0) {
                ft::save_json(data / "integration-result.json", {{"passed", true}, {"stream", true}, {"reload_preserved_history", true}, {"cancellation", true}});
                integration_stage = 5; closing = true; window.hide();
            }
            if(std::chrono::steady_clock::now() > integration_deadline) {
                closing = true; stop_request(); window.hide();
            }
        }
    }

    explicit App(std::filesystem::path directory = ft::data_directory()) : data(std::move(directory)) {
        hardware = ft::detect_hardware();
        Fl::scheme("base");
        Fl::background(7, 10, 18);
        Fl::background2(15, 21, 34);
        Fl::foreground(225, 233, 246);
        Fl::set_font(FL_HELVETICA, "Segoe UI");
        Fl::set_font(FL_HELVETICA_BOLD, "Segoe UI Semibold");
        Fl::set_font(FL_HELVETICA_ITALIC, "Segoe UI");
        Fl::set_boxtype(theme::rounded, theme::surface, 8, 6, 16, 12);
        Fl::scrollbar_size(10);

        window.begin();

        // -------------------------------------------------------------
        // SIDEBAR (x=0, y=0, w=200, h=800)
        // -------------------------------------------------------------
        auto* rail = new Fl_Box(0, 0, 200, 800);
        rail->box(FL_FLAT_BOX); rail->color(theme::sidebar);

        // Logo & Branding
        auto* logo_icon = new Fl_Box(18, 15, 38, 38, "FT");
        logo_icon->box(theme::rounded); logo_icon->color(theme::accent);
        logo_icon->labelcolor(theme::sidebar); logo_icon->labelfont(FL_COURIER_BOLD); logo_icon->labelsize(14);

        auto* brand = new Fl_Box(66, 14, 124, 22, "FreeP100");
        brand->align(FL_ALIGN_LEFT | FL_ALIGN_INSIDE); brand->labelfont(FL_HELVETICA_BOLD);
        brand->labelsize(17); brand->labelcolor(theme::text_bright);

        auto* tagline = new Fl_Box(66, 36, 124, 14, "P100 INFERENCE CONSOLE");
        tagline->align(FL_ALIGN_LEFT | FL_ALIGN_INSIDE); tagline->labelsize(10); tagline->labelcolor(theme::muted);

        auto* workspace_label = new Fl_Box(14, 68, 172, 12, "WORKSPACE");
        workspace_label->align(FL_ALIGN_LEFT | FL_ALIGN_INSIDE); workspace_label->labelfont(FL_COURIER_BOLD);
        workspace_label->labelsize(9); workspace_label->labelcolor(theme::subtle);

        // Navigation (Chat, Models, Tune, Monitor)
        const char* nav_names[] = {"Chat", "Model Vault", "Tuning Lab", "Telemetry"};
        const char* nav_icons[] = {"01", "02", "03", "04"};
        for(int i = 0; i < 4; ++i) {
            navigation[i] = new NavButton(10, 86 + i * 44, 180, 38, nav_names[i], nav_icons[i]);
            navigation[i]->callback([](Fl_Widget* w, void* p) {
                auto* a = static_cast<App*>(p);
                for(int j = 0; j < 4; ++j) if(a->navigation[j] == w) a->navigate(j);
            }, this);
        }

        auto* telemetry_label = new Fl_Box(16, 570, 168, 14, "SYSTEM TELEMETRY");
        telemetry_label->align(FL_ALIGN_LEFT | FL_ALIGN_INSIDE); telemetry_label->labelfont(FL_COURIER_BOLD);
        telemetry_label->labelsize(9); telemetry_label->labelcolor(theme::subtle);

        auto* telemetry_card = new Fl_Box(10, 592, 180, 126);
        telemetry_card->box(theme::rounded); telemetry_card->color(theme::panel);

        // Mini Gauges at Bottom of Sidebar
        vramGaugeLabel = new Fl_Box(20, 604, 160, 14, "VRAM  -");
        vramGaugeLabel->align(FL_ALIGN_LEFT | FL_ALIGN_INSIDE); vramGaugeLabel->labelsize(10); vramGaugeLabel->labelcolor(theme::muted);
        vramGauge = new Fl_Progress(20, 622, 160, 5);
        vramGauge->minimum(0); vramGauge->maximum(100); vramGauge->value(0);
        vramGauge->box(FL_FLAT_BOX); vramGauge->color(theme::panel); vramGauge->selection_color(theme::accent);

        ramGaugeLabel = new Fl_Box(20, 640, 160, 14, "RAM  -");
        ramGaugeLabel->align(FL_ALIGN_LEFT | FL_ALIGN_INSIDE); ramGaugeLabel->labelsize(10); ramGaugeLabel->labelcolor(theme::muted);
        ramGauge = new Fl_Progress(20, 658, 160, 5);
        ramGauge->minimum(0); ramGauge->maximum(100); ramGauge->value(0);
        ramGauge->box(FL_FLAT_BOX); ramGauge->color(theme::panel); ramGauge->selection_color(theme::status_blue);

        chipLine = new Fl_Box(20, 672, 160, 38, "CTX  4096\n0 requests · 0 out");
        chipLine->align(FL_ALIGN_LEFT | FL_ALIGN_INSIDE | FL_ALIGN_WRAP); chipLine->labelsize(10); chipLine->labelcolor(theme::muted);

        auto* local_ws = new Fl_Box(16, 736, 168, 30, "PRIVATE · OFFLINE FIRST\nGPU ACCELERATED");
        local_ws->align(FL_ALIGN_LEFT | FL_ALIGN_INSIDE); local_ws->labelsize(10); local_ws->labelcolor(theme::subtle);

        // -------------------------------------------------------------
        // TOP HEADER BAR (x=200, y=0, w=1000, h=50)
        // -------------------------------------------------------------
        page_title = new Fl_Box(216, 6, 180, 26, "Chat");
        page_title->align(FL_ALIGN_LEFT | FL_ALIGN_INSIDE); page_title->labelfont(FL_HELVETICA_BOLD); page_title->labelsize(18);

        page_subtitle = new Fl_Box(216, 31, 220, 13, "LOCAL INFERENCE SESSION");
        page_subtitle->align(FL_ALIGN_LEFT | FL_ALIGN_INSIDE); page_subtitle->labelfont(FL_COURIER_BOLD);
        page_subtitle->labelsize(9); page_subtitle->labelcolor(theme::subtle);

        capsule = new ModelCapsule(420, 8, 480, 36);
        capsule->action_btn->callback([](Fl_Widget*, void* p) {
            auto* a = static_cast<App*>(p);
            if(a->busy && !a->ready) a->stop_request();
            else if(a->ready) {
                a->process.stop(); a->ready = false; a->set_busy(false);
                a->set_status("Model unloaded.");
            } else a->start();
        }, this);

        capsule_menu = new Fl_Menu_Button(420, 8, 250, 36, "");
        capsule_menu->box(FL_NO_BOX);

        top_params_btn = new ActionBtn(1070, 10, 114, 30, "Parameters", ActionBtn::Secondary);
        top_params_btn->labelsize(11);
        top_params_btn->callback([](Fl_Widget*, void* p) { static_cast<App*>(p)->toggle_inspector(); }, this);

        top_server_chip = new Fl_Box(960, 10, 100, 30, "○ Server stopped");
        top_server_chip->align(FL_ALIGN_RIGHT | FL_ALIGN_INSIDE);
        top_server_chip->labelsize(11);
        top_server_chip->labelcolor(theme::muted);

        // Alias for compatibility
        load = capsule->action_btn;
        unload = capsule->action_btn;

        // Hidden model picker for backward-compatibility with tests
        model_picker = new Choice(0, 0, 10, 10);
        model_picker->hide();

        // -------------------------------------------------------------
        // MAIN TABS CONTAINER (x=200, y=50, w=1000, h=722)
        // -------------------------------------------------------------
        tabs = new Fl_Group(200, 50, 1000, 722);

        // =============================================================
        // TAB 0: CHAT (Workspace + Floating Composer + Inspector)
        // =============================================================
        auto* chat_group = new Fl_Group(200, 50, 1000, 722);

        chat_stream = new ChatDisplay(212, 58, 976, 540);
        chat_view = chat_stream; // compatibility handle

        composer_card = new ComposerCard(216, 610, 968, 100);
        prompt = composer_card->input;
        prompt->on_submit = [this] { chat(); };

        send = composer_card->send_btn;
        send->callback([](Fl_Widget*, void* p) { static_cast<App*>(p)->chat(); }, this);

        cancel = composer_card->stop_btn;
        cancel->callback([](Fl_Widget*, void* p) { static_cast<App*>(p)->stop_request(); }, this);

        fresh = composer_card->new_chat_btn;
        fresh->callback([](Fl_Widget*, void* p) { static_cast<App*>(p)->new_chat(); }, this);

        chat_inspector = new ChatInspector(920, 50, 280, 722);
        chat_inspector->close_btn->callback([](Fl_Widget*, void* p) { static_cast<App*>(p)->toggle_inspector(); }, this);

        // Wire up inspector controls
        chat_inspector->apply_sys_btn->callback([](Fl_Widget*, void* p) {
            auto* a = static_cast<App*>(p);
            a->config["system"] = a->chat_inspector->sys_prompt->value();
            if(a->fields.count("system")) a->fields["system"]->value(a->chat_inspector->sys_prompt->value());
            a->save();
            a->set_status("System prompt updated.");
        }, this);

        chat_inspector->preset_choice->callback([](Fl_Widget* w, void* p) {
            auto* a = static_cast<App*>(p);
            int idx = static_cast<Fl_Choice*>(w)->value();
            const char* presets[] = {
                "You are a helpful, respectful, and honest assistant.",
                "You are an expert software engineer. Provide high-performance, well-structured, production-ready code with concise explanations.",
                "Answer directly and concisely with zero fluff or conversational filler.",
                "You are a creative writer and thought partner. Explore novel perspectives and imaginative solutions.",
                ""
            };
            if(idx >= 0 && idx < 4) {
                a->chat_inspector->sys_prompt->value(presets[idx]);
                a->config["system"] = presets[idx];
                if(a->fields.count("system")) a->fields["system"]->value(presets[idx]);
                a->save();
            }
        }, this);

        chat_inspector->temp_slider->callback([](Fl_Widget* w, void* p) {
            auto* a = static_cast<App*>(p);
            double v = static_cast<Fl_Value_Slider*>(w)->value();
            if(a->sliders.count("temperature")) a->sliders["temperature"]->value(v);
            a->save();
        }, this);

        chat_inspector->tokens_slider->callback([](Fl_Widget* w, void* p) {
            auto* a = static_cast<App*>(p);
            double v = static_cast<Fl_Value_Slider*>(w)->value();
            if(a->sliders.count("max_tokens")) a->sliders["max_tokens"]->value(v);
            a->save();
        }, this);

        chat_inspector->gpu_slider->callback([](Fl_Widget* w, void* p) {
            auto* a = static_cast<App*>(p);
            double v = static_cast<Fl_Value_Slider*>(w)->value();
            if(a->sliders.count("gpu_layers")) a->sliders["gpu_layers"]->value(v);
            a->edited_cb(nullptr, a);
        }, this);

        chat_inspector->thinking_switch->callback([](Fl_Widget* w, void* p) {
            auto* a = static_cast<App*>(p);
            bool v = static_cast<Fl_Check_Button*>(w)->value();
            if(a->toggles.count("thinking")) a->toggles["thinking"]->value(v);
            a->save();
        }, this);

        chat_inspector->hide();

        // Loading card overlay
        loading_card = new LoadingCard(360, 180, 520, 120);
        loading_card->cancel_btn->callback([](Fl_Widget*, void* p) { static_cast<App*>(p)->stop_request(); }, this);
        loading_card->hide();

        chat_group->resizable(chat_stream);
        chat_group->end();
        pages[0] = chat_group;

        // =============================================================
        // TAB 1: MODELS (Split View)
        // =============================================================
        auto* connection = new Fl_Group(200, 50, 1000, 722);

        auto* dir_lbl = new Fl_Box(216, 58, 480, 18, "YOUR MODEL FOLDER  /  Scans all subfolders");
        dir_lbl->align(FL_ALIGN_LEFT | FL_ALIGN_INSIDE); dir_lbl->labelsize(11); dir_lbl->labelcolor(theme::muted);
        dir_lbl->labelfont(FL_HELVETICA_BOLD);

        auto* mf = new Fl_Input(216, 78, 480, 32);
        mf->box(theme::rounded); mf->color(theme::panel); mf->textcolor(theme::text);
        mf->callback(edited_cb, this);
        fields["model_folder"] = mf;

        auto* choose_folder = new ActionBtn(706, 78, 120, 32, "Choose folder");
        choose_folder->callback([](Fl_Widget*, void* p) {
            Fl_Native_File_Chooser c; c.title("Select Model Folder");
            c.type(Fl_Native_File_Chooser::BROWSE_DIRECTORY);
            if(c.show() == 0) {
                auto* a = static_cast<App*>(p);
                a->fields.at("model_folder")->value(c.filename());
                a->scan_folder();
            }
        }, this);

        auto* rescan = new ActionBtn(836, 78, 80, 32, "Rescan");
        rescan->callback([](Fl_Widget*, void* p) { static_cast<App*>(p)->scan_folder(); }, this);

        library_info = new Fl_Box(216, 118, 480, 22, "0 models  ·  choose a folder to build your local vault");
        library_info->align(FL_ALIGN_LEFT | FL_ALIGN_INSIDE); library_info->labelsize(10);
        library_info->labelfont(FL_COURIER_BOLD); library_info->labelcolor(theme::muted);

        model_list = new Fl_Hold_Browser(216, 142, 480, 458);
        model_list->box(theme::rounded); model_list->color(theme::panel);
        model_list->textcolor(theme::text); model_list->textsize(13);
        model_list->callback([](Fl_Widget* w, void* p) {
            auto* b = static_cast<Fl_Hold_Browser*>(w);
            if(b->value() > 0) static_cast<App*>(p)->select_model(b->value() - 1);
        }, this);

        auto* details_card = new Fl_Group(706, 142, 480, 458);
        details_card->box(theme::rounded); details_card->color(theme::panel);

        model_details_title = new Fl_Box(722, 160, 448, 26, "Select a model");
        model_details_title->align(FL_ALIGN_LEFT | FL_ALIGN_INSIDE);
        model_details_title->labelfont(FL_HELVETICA_BOLD); model_details_title->labelsize(16);

        model_details_meta = new Fl_Box(722, 194, 448, 92, "Choose a GGUF to inspect its architecture, context capacity, file size, layer count, and projector pairing.");
        model_details_meta->align(FL_ALIGN_LEFT | FL_ALIGN_TOP | FL_ALIGN_INSIDE | FL_ALIGN_WRAP);
        model_details_meta->labelsize(12); model_details_meta->labelcolor(theme::muted);

        model_fit_badge = new Fl_Box(722, 294, 438, 32, "AUTO PROFILE  ·  hardware-aware placement will be calculated on selection");
        model_fit_badge->box(theme::rounded); model_fit_badge->color(theme::accent_dark);
        model_fit_badge->align(FL_ALIGN_LEFT | FL_ALIGN_INSIDE); model_fit_badge->labelsize(10); model_fit_badge->labelcolor(theme::accent);

        auto* proj_lbl = new Fl_Box(722, 340, 448, 18, "VISION PROJECTOR  /  OPTIONAL PAIRING");
        proj_lbl->align(FL_ALIGN_LEFT | FL_ALIGN_INSIDE); proj_lbl->labelsize(11); proj_lbl->labelcolor(theme::muted);
        proj_lbl->labelfont(FL_HELVETICA_BOLD);

        auto* mm = new Fl_Input(722, 362, 350, 32);
        mm->box(theme::rounded); mm->color(fl_rgb_color(16, 20, 26)); mm->textcolor(theme::text);
        mm->callback(edited_cb, this);
        fields["mmproj"] = mm;

        auto* choose_proj = new ActionBtn(1082, 362, 78, 32, "Browse");
        choose_proj->callback([](Fl_Widget*, void* p) { static_cast<App*>(p)->choose("mmproj", false); }, this);

        details_load_btn = new ActionBtn(722, 416, 140, 38, "Load Model", ActionBtn::Primary);
        details_load_btn->callback(load_cb, this);

        details_unload_btn = new ActionBtn(872, 416, 90, 38, "Unload", ActionBtn::Secondary);
        details_unload_btn->callback([](Fl_Widget*, void* p) {
            auto* a = static_cast<App*>(p);
            a->process.stop(); a->ready = false; a->set_busy(false);
            a->set_status("Model unloaded. Conversation retained.");
        }, this);

        details_card->end();

        auto* m_lbl = new Fl_Box(216, 610, 480, 18, "SELECTED MODEL PATH");
        m_lbl->align(FL_ALIGN_LEFT | FL_ALIGN_INSIDE); m_lbl->labelsize(11); m_lbl->labelcolor(theme::muted);
        m_lbl->labelfont(FL_HELVETICA_BOLD);

        auto* m_input = new Fl_Input(216, 630, 390, 32);
        m_input->box(theme::rounded); m_input->color(theme::panel); m_input->textcolor(theme::text);
        m_input->callback(edited_cb, this);
        fields["model"] = m_input;

        auto* choose_model_btn = new ActionBtn(616, 630, 100, 32, "Add GGUF");
        choose_model_btn->callback([](Fl_Widget*, void* p) { static_cast<App*>(p)->choose("model", false); }, this);

        connection->resizable(model_list);
        connection->end();
        pages[1] = connection;

        // =============================================================
        // TAB 2: TUNE (Settings + Live Memory Plan Side-by-Side)
        // =============================================================
        auto* settings_group = new Fl_Group(200, 50, 1000, 722);

        auto* rec_auto = new Switch(216, 56, 320, 28, "FreeToken Auto  /  Recommend on selection");
        rec_auto->callback(edited_cb, this);
        toggles["recommend"] = rec_auto;

        auto* detect_hw = new ActionBtn(546, 56, 120, 28, "Detect hardware");
        detect_hw->callback([](Fl_Widget*, void* p) {
            auto* a = static_cast<App*>(p);
            a->hardware = ft::detect_hardware();
            a->show_config();
            a->set_status("Hardware re-detected.");
        }, this);

        auto* restore_rec = new ActionBtn(676, 56, 120, 28, "Restore defaults");
        restore_rec->callback([](Fl_Widget*, void* p) { static_cast<App*>(p)->restore_profile(); }, this);

        profile_info = new Fl_Box(216, 92, 580, 18, "RECOMMENDED  /  Hardware-based starting point");
        profile_info->align(FL_ALIGN_LEFT | FL_ALIGN_INSIDE);
        profile_info->labelfont(FL_HELVETICA_BOLD); profile_info->labelsize(11); profile_info->labelcolor(theme::accent);

        hardware_info = new Fl_Box(216, 112, 580, 18, "Detecting hardware...");
        hardware_info->align(FL_ALIGN_LEFT | FL_ALIGN_INSIDE); hardware_info->labelsize(11); hardware_info->labelcolor(theme::muted);

        settings_scroll = new Fl_Scroll(216, 138, 480, 560);
        settings_scroll->box(FL_NO_BOX);

        auto add_input = [&](Fl_Group* parent, int y, const char* key, const char* label, const char* desc, bool numeric) {
            parent->begin();
            auto* title = new Fl_Box(220, y, 200, 18, label);
            title->align(FL_ALIGN_LEFT | FL_ALIGN_INSIDE); title->labelsize(12); title->labelcolor(theme::text);
            auto* help = new Fl_Box(220, y + 18, 200, 24, desc);
            help->align(FL_ALIGN_LEFT | FL_ALIGN_INSIDE | FL_ALIGN_WRAP); help->labelsize(10); help->labelcolor(theme::muted);
            auto* in = new Fl_Input(430, y + 6, 250, 28);
            in->box(theme::rounded); in->color(fl_rgb_color(16, 20, 26)); in->textcolor(theme::text);
            if(numeric) in->type(FL_INT_INPUT);
            in->callback(edited_cb, this);
            fields[key] = in;
            parent->end();
        };

        auto add_slider = [&](Fl_Group* parent, int y, const char* key, const char* label, const char* desc, double min_v, double max_v, double step, bool all_at_max = false) {
            parent->begin();
            auto* title = new Fl_Box(220, y, 200, 18, label);
            title->align(FL_ALIGN_LEFT | FL_ALIGN_INSIDE); title->labelsize(12); title->labelcolor(theme::text);
            auto* help = new Fl_Box(220, y + 18, 200, 24, desc);
            help->align(FL_ALIGN_LEFT | FL_ALIGN_INSIDE | FL_ALIGN_WRAP); help->labelsize(10); help->labelcolor(theme::muted);
            auto* sl = new Slider(430, y + 8, 250, 24);
            sl->bounds(min_v, max_v); sl->step(step); sl->all_at_max = all_at_max;
            sl->callback(edited_cb, this);
            sliders[key] = sl;
            parent->end();
        };

        auto add_toggle = [&](Fl_Group* parent, int y, const char* key, const char* label, const char* desc) {
            parent->begin();
            auto* help = new Fl_Box(220, y + 20, 460, 18, desc);
            help->align(FL_ALIGN_LEFT | FL_ALIGN_INSIDE); help->labelsize(10); help->labelcolor(theme::muted);
            auto* sw = new Switch(220, y, 460, 24, label);
            sw->callback(edited_cb, this);
            toggles[key] = sw;
            parent->end();
        };

        auto add_choice = [&](Fl_Group* parent, int y, const char* key, const char* label, const char* desc, const std::vector<std::string>& opts) {
            parent->begin();
            auto* title = new Fl_Box(220, y, 200, 18, label);
            title->align(FL_ALIGN_LEFT | FL_ALIGN_INSIDE); title->labelsize(12); title->labelcolor(theme::text);
            auto* help = new Fl_Box(220, y + 18, 200, 24, desc);
            help->align(FL_ALIGN_LEFT | FL_ALIGN_INSIDE | FL_ALIGN_WRAP); help->labelsize(10); help->labelcolor(theme::muted);
            auto* ch = new Choice(430, y + 6, 250, 28);
            for(auto& o : opts) ch->add(o.c_str());
            choice_values[key] = opts;
            ch->callback(edited_cb, this);
            choices[key] = ch;
            parent->end();
        };

        int cur_y = 138;
        auto add_section = [&](const char* title, int body_h, auto populate) {
            auto* disc = new Disclosure(216, cur_y, 480, 36, title);
            auto* body = new Fl_Group(216, cur_y + 36, 480, body_h);
            body->box(FL_NO_BOX);
            populate(body);
            body->end();
            sections.push_back({disc, body, body_h, sections.empty()});
            if(!sections.back().open) { disc->expanded = false; body->hide(); }
            else { disc->expanded = true; body->show(); }
            disc->callback([](Fl_Widget* w, void* p) {
                auto* a = static_cast<App*>(p);
                for(auto& s : a->sections) if(s.header == w) {
                    s.open = !s.open;
                    static_cast<Disclosure*>(s.header)->expanded = s.open;
                    if(s.open) s.body->show(); else s.body->hide();
                }
                int y = 138;
                for(auto& s : a->sections) {
                    s.header->position(216, y); y += 36;
                    if(s.open) { s.body->position(216, y); y += s.height + 8; }
                }
                a->settings_scroll->init_sizes();
                a->window.redraw();
            }, this);
            cur_y += 36 + (sections.back().open ? body_h + 8 : 0);
        };

        add_section("Context and GPU offload", 230, [&](Fl_Group* b) {
            add_input(b, cur_y + 40, "ctx", "Context tokens", "Requires reload. Total context capacity; custom MoE path uses one slot.", true);
            add_slider(b, cur_y + 90, "gpu_layers", "GPU layers", "Attention/shared layer placement. 0 keeps these on CPU.", 0, 99, 1, true);
            add_slider(b, cur_y + 140, "threads", "Decode threads", "CPU worker threads. Start near physical core count.", 1, 32, 1);
            add_input(b, cur_y + 190, "batch", "Batch tokens", "Logical maximum prompt batch.", true);
            add_input(b, cur_y + 240, "ubatch", "Microbatch tokens", "Physical batch; cannot exceed logical batch.", true);
        });

        add_section("Attention and memory", 200, [&](Fl_Group* b) {
            add_choice(b, cur_y + 40, "flash", "FlashAttention", "Enable kernels if GPU architecture permits.", {"off", "on"});
            add_choice(b, cur_y + 90, "cache_type_k", "KV cache K type", "KV cache quantization format for Keys.", {"f16", "q8_0", "q4_0"});
            add_choice(b, cur_y + 140, "cache_type_v", "KV cache V type", "KV cache quantization format for Values.", {"f16", "q8_0", "q4_0"});
            add_input(b, cur_y + 190, "reserve", "VRAM headroom (MiB)", "Buffer left free for display and CUDA graphs.", true);
        });

        add_section("Inference and sampling", 260, [&](Fl_Group* b) {
            add_input(b, cur_y + 40, "system", "System prompt", "Instructions applied to this conversation.", false);
            add_toggle(b, cur_y + 90, "thinking", "Enable thinking", "For models supporting thinking tags (<thought>).");
            add_slider(b, cur_y + 130, "temperature", "Temperature", "Lower is focused. Higher is more varied.", 0.0, 2.0, 0.05);
            add_input(b, cur_y + 175, "max_tokens", "Response length", "Maximum generated tokens, including reasoning.", true);
            add_input(b, cur_y + 225, "top_k", "Top K", "Limit candidate tokens; 0 disables the limit.", true);
            add_slider(b, cur_y + 275, "top_p", "Top P", "Cumulative probability of candidate tokens.", 0.0, 1.0, 0.05);
            add_slider(b, cur_y + 325, "min_p", "Min P", "Exclude tokens unlikely relative to the best candidate.", 0.0, 1.0, 0.01);
            add_slider(b, cur_y + 375, "repeat_penalty", "Repeat penalty", "Penalize repetitive token generation.", 1.0, 2.0, 0.05);
        });

        add_section("FreeToken expert caching", 180, [&](Fl_Group* b) {
            add_toggle(b, cur_y + 40, "expert_enabled", "Enable expert caching", "Maintains high-priority MoE weights in fast cache.");
            add_input(b, cur_y + 80, "gpu_cache", "GPU expert cache (MiB)", "VRAM dedicated to expert layers.", true);
            add_input(b, cur_y + 130, "cpu_cache", "CPU expert cache (MiB)", "System RAM dedicated to expert layers.", true);
            add_choice(b, cur_y + 180, "policy", "Eviction policy", "Algorithm for expert cache management.", {"mru", "lru"});
        });

        add_section("Advanced execution", 220, [&](Fl_Group* b) {
            add_choice(b, cur_y + 40, "graphs", "CUDA graphs", "Enables capture & replay. P100 disables replay by default.", {"auto", "off", "on"});
            add_toggle(b, cur_y + 90, "fast", "Fast MoE", "Accelerated MoE tensor operations.");
            add_toggle(b, cur_y + 130, "pipeline", "Pipelined decode", "Overlap GPU tensor compute with CPU scheduling.");
            add_toggle(b, cur_y + 170, "no_repack", "No repack", "Avoid repacking weights in system memory.");
            add_toggle(b, cur_y + 210, "no_warmup", "Skip warmup", "Skip dummy prompt warmup during startup.");
        });

        settings_scroll->end();

        // Right side: Interactive Live Memory Plan
        plan_panel = new Fl_Group(716, 138, 460, 560);
        plan_panel->box(theme::rounded); plan_panel->color(theme::panel);

        auto* plan_title = new Fl_Box(732, 150, 428, 22, "LIVE MEMORY PLAN");
        plan_title->align(FL_ALIGN_LEFT | FL_ALIGN_INSIDE);
        plan_title->labelfont(FL_HELVETICA_BOLD); plan_title->labelsize(13); plan_title->labelcolor(theme::text_bright);

        auto* plan_subtitle = new Fl_Box(732, 172, 428, 16, "Updates in real-time as sliders are adjusted:");
        plan_subtitle->align(FL_ALIGN_LEFT | FL_ALIGN_INSIDE); plan_subtitle->labelsize(10); plan_subtitle->labelcolor(theme::muted);

        plan_model_size = new Fl_Box(732, 196, 428, 18, "Model: - GiB (- GPU layers)");
        plan_model_size->align(FL_ALIGN_LEFT | FL_ALIGN_INSIDE); plan_model_size->labelsize(12); plan_model_size->labelcolor(theme::text);

        plan_gpu_weights = new Fl_Box(732, 218, 428, 18, "GPU Weights: - GiB");
        plan_gpu_weights->align(FL_ALIGN_LEFT | FL_ALIGN_INSIDE); plan_gpu_weights->labelsize(12); plan_gpu_weights->labelcolor(theme::accent);

        plan_kv_cache = new Fl_Box(732, 240, 428, 18, "KV Cache: - GiB (- ctx)");
        plan_kv_cache->align(FL_ALIGN_LEFT | FL_ALIGN_INSIDE); plan_kv_cache->labelsize(12); plan_kv_cache->labelcolor(theme::status_blue);

        plan_ram_weights = new Fl_Box(732, 262, 428, 18, "RAM Weights: - GiB");
        plan_ram_weights->align(FL_ALIGN_LEFT | FL_ALIGN_INSIDE); plan_ram_weights->labelsize(12); plan_ram_weights->labelcolor(theme::muted);

        plan_vram_total = new Fl_Box(732, 290, 428, 18, "Estimated VRAM: - GiB / - GiB");
        plan_vram_total->align(FL_ALIGN_LEFT | FL_ALIGN_INSIDE); plan_vram_total->labelsize(12);
        plan_vram_total->labelfont(FL_HELVETICA_BOLD); plan_vram_total->labelcolor(theme::text_bright);

        plan_vram_gauge = new Fl_Progress(732, 312, 428, 8);
        plan_vram_gauge->box(FL_FLAT_BOX); plan_vram_gauge->color(fl_rgb_color(16, 20, 26));
        plan_vram_gauge->selection_color(theme::accent); plan_vram_gauge->minimum(0); plan_vram_gauge->maximum(100); plan_vram_gauge->value(0);

        plan_guard_notice = new Fl_Box(732, 334, 428, 16, "CUDA graphs: Auto (follows backend)");
        plan_guard_notice->align(FL_ALIGN_LEFT | FL_ALIGN_INSIDE); plan_guard_notice->labelsize(10); plan_guard_notice->labelcolor(theme::muted);

        review_view = new Fl_Text_Display(732, 356, 428, 280);
        review_view->buffer(review);
        review_view->box(theme::rounded); review_view->color(fl_rgb_color(16, 20, 26));
        review_view->textcolor(theme::text); review_view->textsize(11);

        auto* copy_plan = new ActionBtn(732, 646, 110, 28, "Copy Plan");
        copy_plan->callback([](Fl_Widget*, void* p) {
            auto* a = static_cast<App*>(p);
            std::string text = a->review.text();
            Fl::copy(text.c_str(), static_cast<int>(text.size()), 1);
            a->set_status("Memory plan copied to clipboard.");
        }, this);

        plan_panel->end();

        settings_group->resizable(settings_scroll);
        settings_group->end();
        pages[2] = settings_group;

        // =============================================================
        // TAB 3: MONITOR (Hero Cards + Server Settings + Streaming Log)
        // =============================================================
        auto* log_group = new Fl_Group(200, 50, 1000, 722);

        auto make_hero = [](int x, int y, int w, int h, const char* title, const char* initial_val, Fl_Box*& val_box) {
            auto* g = new Fl_Group(x, y, w, h);
            g->box(theme::rounded); g->color(theme::panel);
            auto* lbl = new Fl_Box(x + 14, y + 10, w - 28, 16, title);
            lbl->align(FL_ALIGN_LEFT | FL_ALIGN_INSIDE); lbl->labelsize(10); lbl->labelcolor(theme::muted);
            lbl->labelfont(FL_HELVETICA_BOLD);
            val_box = new Fl_Box(x + 14, y + 28, w - 28, 28, initial_val);
            val_box->align(FL_ALIGN_LEFT | FL_ALIGN_INSIDE); val_box->labelsize(15); val_box->labelfont(FL_HELVETICA_BOLD);
            val_box->labelcolor(theme::text_bright);
            g->end();
            return g;
        };

        make_hero(216, 58, 230, 68, "ACTIVE MODEL", "None · stopped", hero_model);
        make_hero(458, 58, 160, 68, "DECODE SPEED", "— tok/s", hero_tps);
        make_hero(630, 58, 190, 68, "FIRST TOKEN", "— ms", hero_ttft);
        make_hero(832, 58, 230, 68, "SESSION TRAFFIC", "0 in / 0 out", hero_tokens);

        auto make_metric = [](int x, int y, int w, const char* title, const char* initial, Fl_Color signal, Fl_Box*& value) {
            auto* g = new Fl_Group(x, y, w, 64);
            g->box(theme::rounded); g->color(fl_rgb_color(11, 17, 28));
            auto* rail = new Fl_Box(x + 12, y + 12, 3, 40);
            rail->box(FL_FLAT_BOX); rail->color(signal);
            auto* lbl = new Fl_Box(x + 24, y + 9, w - 36, 17, title);
            lbl->align(FL_ALIGN_LEFT | FL_ALIGN_INSIDE); lbl->labelsize(9); lbl->labelfont(FL_COURIER_BOLD); lbl->labelcolor(theme::muted);
            value = new Fl_Box(x + 24, y + 27, w - 36, 24, initial);
            value->align(FL_ALIGN_LEFT | FL_ALIGN_INSIDE); value->labelsize(12); value->labelfont(FL_HELVETICA_BOLD); value->labelcolor(theme::text);
            g->end();
        };
        make_metric(216, 136, 230, "GPU MEMORY", "Waiting for telemetry", theme::accent, metric_vram);
        make_metric(458, 136, 230, "HOST MEMORY", "Waiting for telemetry", theme::status_blue, metric_ram);
        make_metric(700, 136, 230, "CONTEXT & SLOTS", "4096 tokens · idle", theme::violet, metric_context);
        make_metric(942, 136, 234, "REQUEST TOTALS", "0 req · 0 tokens", theme::status_amber, metric_session);

        auto* log_label = new Fl_Box(216, 208, 420, 16, "ENGINE EVENT STREAM  /  LIVE SERVER OUTPUT");
        log_label->align(FL_ALIGN_LEFT | FL_ALIGN_INSIDE); log_label->labelsize(10); log_label->labelfont(FL_COURIER_BOLD); log_label->labelcolor(theme::muted);

        auto* srv_lbl = new Fl_Box(216, 696, 500, 14, "RUNTIME  /  LLAMA-SERVER EXECUTABLE");
        srv_lbl->align(FL_ALIGN_LEFT | FL_ALIGN_INSIDE); srv_lbl->labelsize(11); srv_lbl->labelcolor(theme::muted);
        srv_lbl->labelfont(FL_HELVETICA_BOLD);

        auto* srv_in = new Fl_Input(216, 714, 520, 32);
        srv_in->box(theme::rounded); srv_in->color(theme::panel); srv_in->textcolor(theme::text);
        srv_in->callback(edited_cb, this);
        fields["server"] = srv_in;

        auto* choose_server_btn = new ActionBtn(746, 714, 104, 32, "Browse");
        choose_server_btn->callback([](Fl_Widget*, void* p) { static_cast<App*>(p)->choose("server", true); }, this);

        auto* port_lbl = new Fl_Box(864, 696, 90, 14, "LOCAL PORT");
        port_lbl->align(FL_ALIGN_LEFT | FL_ALIGN_INSIDE); port_lbl->labelsize(11); port_lbl->labelcolor(theme::muted);
        port_lbl->labelfont(FL_HELVETICA_BOLD);

        auto* port_in = new Fl_Input(864, 714, 88, 32);
        port_in->box(theme::rounded); port_in->color(theme::panel); port_in->textcolor(theme::text);
        port_in->type(FL_INT_INPUT); port_in->callback(edited_cb, this);
        fields["port"] = port_in;

        auto* save_srv = new ActionBtn(962, 714, 108, 32, "Save runtime");
        save_srv->callback([](Fl_Widget*, void* p) {
            auto* a = static_cast<App*>(p); a->save(); a->set_status("Server settings saved.");
        }, this);

        log_view = new Fl_Text_Display(216, 226, 960, 468);
        log_view->buffer(logs);
        log_view->box(theme::rounded); log_view->color(fl_rgb_color(12, 15, 20));
        log_view->textcolor(theme::text); log_view->textsize(11);

        auto* refresh_btn = new ActionBtn(1048, 202, 62, 24, "Refresh");
        refresh_btn->callback([](Fl_Widget*, void* p) { static_cast<App*>(p)->refresh_log(); }, this);

        auto* clear_btn = new ActionBtn(1116, 202, 60, 24, "Clear");
        clear_btn->callback([](Fl_Widget*, void* p) { static_cast<App*>(p)->logs.text(""); }, this);

        log_group->resizable(log_view);
        log_group->end();
        pages[3] = log_group;

        tabs->end();

        // -------------------------------------------------------------
        // BOTTOM STATUS BAR (x=200, y=772, w=1000, h=28)
        // -------------------------------------------------------------
        status = new Fl_Box(216, 774, 600, 24, "Ready");
        status->align(FL_ALIGN_LEFT | FL_ALIGN_INSIDE);
        status->labelsize(11); status->labelcolor(theme::muted);

        bottom_ctx_chip = new Fl_Box(820, 774, 360, 24, "ctx - · 0 reqs");
        bottom_ctx_chip->align(FL_ALIGN_RIGHT | FL_ALIGN_INSIDE);
        bottom_ctx_chip->labelsize(11); bottom_ctx_chip->labelcolor(theme::muted);

        window.end();
        window.resizable(tabs);

        window.callback([](Fl_Widget*, void* p) {
            auto* a = static_cast<App*>(p);
            a->closing = true;
            a->stop_request();
            a->process.stop();
            if(a->telem.joinable()) a->telem.join();
            if(a->worker.joinable()) a->worker.join();
            if(a->scanner.joinable()) a->scanner.join();
            a->window.hide();
        }, this);

        try { config = ft::read_json(data / "settings.json", ft::defaults()); }
        catch(...) { config = ft::defaults(); }

        try {
            messages = ft::read_json(data / "conversation.json", ft::json::array());
            if(chat_stream) {
                for(auto& m : messages) {
                    if(m.value("role", "") == "user") chat_stream->add_user_message(m.value("content", ""));
                    else {
                        chat_stream->start_assistant_message(active_model_display_name());
                        chat_stream->append_token(m.value("content", ""));
                        chat_stream->finish_assistant_message("");
                    }
                }
            }
        } catch(...) { messages = ft::json::array(); }

        show_config();
        navigate(0);
        layout_sections();
        set_busy(false);
    }

    ~App() {
        closing = true;
        Fl::remove_timeout(poll_cb, this);
        stop_request();
        if(worker.joinable()) worker.join();
        if(scanner.joinable()) scanner.join();
        if(telem.joinable()) telem.join();
        process.stop();
        if(review_view) review_view->buffer(nullptr);
        if(log_view) log_view->buffer(nullptr);
    }

    int visual_check(const std::filesystem::path& output) {
        std::filesystem::create_directories(output);
        window.set_visible();
        auto capture = [&](const std::string& name) {
            window.clear_damage(FL_DAMAGE_ALL);
            Fl_Image_Surface surf(window.w(), window.h());
            Fl_Surface_Device::push_current(&surf);
            surf.draw(&window);
            auto* img = surf.image();
            Fl_Surface_Device::pop_current();
            auto result = fl_write_png((output / name).u8string().c_str(), img);
            delete img;
            if(result != 0) throw std::runtime_error("Could not write UI preview");
        };

        navigate(1); capture("library.png");
        navigate(2); capture("settings.png");
        if(!config["recommended"].empty()) {
            auto baseline = config["recommended"];
            fields.at("ctx")->value("1024");
            profile_state();
            if(std::string(profile_info->label()).find("CUSTOMIZED") == std::string::npos) throw std::runtime_error("Override was not detected");
            save();
            restore_profile();
            if(values()["ctx"] != baseline["ctx"] || config["recommended"] != baseline) throw std::runtime_error("Restore changed the recommendation");
        }
        for(auto& section : sections) section.open = false;
        if(sections.size() > 2) sections[2].open = true;
        layout_sections();
        capture("inference.png");

        window.resize(0, 0, 1440, 900);
        layout_sections();
        capture("resized.png");

        navigate(3); capture("monitor.png");

        navigate(0); capture("chat.png");

        // Capture chat with LM Studio message cards
        if(chat_stream) {
            chat_stream->clear_messages();
            chat_stream->add_user_message("Hello, how fast can you run on the local GPU?");
            chat_stream->start_assistant_message("gemma-4-26B_q4_0-it");
            chat_stream->append_token("Hello! With FreeToken offload and CUDA acceleration, I generate at high speed with live per-token telemetry.");
            chat_stream->finish_assistant_message("[ ⚡ 20.8 tok/s  ·  ⏱ TTFT 284 ms  ·  📊 318 tokens  ·  completed ]");
        }
        capture("chat_messages.png");

        // Capture staged model loading card
        if(loading_card) {
            std::string mname = active_model_display_name();
            loading_card->title_box->copy_label(("Loading Model: " + mname).c_str());
            loading_card->stage_box->copy_label("Allocating KV cache & offloading 30 layers to GPU (68%)...");
            loading_card->progress_bar->value(68);
            loading_card->show();
            window.redraw();
            capture("chat_loading.png");
            loading_card->hide();
            window.redraw();
        }

        // Capture with in-chat parameter inspector open
        set_inspector(true);
        capture("chat_sysprompt.png");
        set_inspector(false);

        ft::save_json(output / "verification.json", {{"passed", true}, {"model_count", library.size()}, {"override_restore", !config["recommended"].empty()}});
        return 0;
    }

    int run(bool smoke, const std::string& test_server = "", const std::string& real_model = "") {
        window.show();
        Fl::add_timeout(0.1, poll_cb, this);
        telem = std::thread([this] { telemetry_loop(); });
        if(test_server.empty() && std::string(fields.at("model_folder")->value()).size()) scan_folder();
#ifdef _WIN32
        BOOL dark = TRUE;
        DwmSetWindowAttribute(fl_xid(&window), 20, &dark, sizeof(dark));
#endif
        if(!test_server.empty()) {
            integration = true; real_test = !real_model.empty();
            integration_deadline = std::chrono::steady_clock::now() + std::chrono::seconds(real_test ? 300 : 20);
            config = ft::defaults(); config["server"] = test_server;
            messages = ft::json::array(); transcript.text(""); style_buf.text("");
            if(chat_stream) chat_stream->clear_messages();
            std::filesystem::create_directories(data);
            auto model = data / "fixture.gguf";
            std::ofstream(model) << "test fixture";
            config["model"] = model.u8string();
            if(real_test) {
                config["model"] = real_model; config["gpu_layers"] = "99"; config["gpu_cache"] = "3060";
                config["fast"] = "1"; config["pipeline"] = "1"; config["ctx"] = "512"; config["batch"] = "128";
                config["ubatch"] = "8"; config["threads"] = "8"; config["threads_batch"] = "8";
                config["flash"] = "off"; config["no_repack"] = "1"; config["no_warmup"] = "1";
                config["max_tokens"] = "32"; config["temperature"] = "0";
            }
            httplib::Server port_probe;
            auto port = port_probe.bind_to_any_port("127.0.0.1");
            port_probe.stop();
            config["port"] = std::to_string(port);
            show_config();
            start();
        }
        if(smoke) Fl::add_timeout(2.0, [](void* p) {
            auto* a = static_cast<App*>(p); a->closing = true; a->window.hide();
        }, this);
        int result = Fl::run();
        return integration ? (integration_stage == 5 ? 0 : 1) : result;
    }
};

int main(int argc, char** argv) {
    try {
        Fl::args_to_utf8(argc, argv);
        if(argc == 3 && std::string(argv[1]) == "--visual-check") {
            auto output = std::filesystem::absolute(std::filesystem::u8path(argv[2]));
            App app(output / "settings");
            return app.visual_check(output);
        }
        if(argc == 4 && std::string(argv[1]) == "--integration-test") {
            App app(std::filesystem::u8path(argv[3]));
            return app.run(false, argv[2]);
        }
        if(argc == 5 && std::string(argv[1]) == "--model-smoke") {
            App app(std::filesystem::u8path(argv[4]));
            return app.run(false, argv[2], argv[3]);
        }
        App app;
        return app.run(argc > 1 && std::string(argv[1]) == "--smoke-test");
    }
    catch(const std::exception& e) {
        fl_alert("FreeP100 Desktop: %s", e.what());
        return 1;
    }
}
