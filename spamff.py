import flet as ft
def main(page: ft.Page):
    page.theme_mode=ft.ThemeMode.DARK
    page.title="Spam Email Detection"
    page.bgcolor="#FCFCFC"
    page.update()
    email=ft.TextField(value=None,label="Enter Your Email",multiline=True,min_lines=5,expand=1,color="black",cursor_color="black")
    r=ft.Text(value="",color=ft.Colors.TRANSPARENT,size=22,weight=ft.FontWeight.BOLD)
    sb=ft.ProgressBar(value=0.0,expand=1,color="#FF0202")
    hb=ft.ProgressBar(value=0.0,expand=1,color="#08D220")
    ham=0.2
    spam=0.8
    def chemail():
        if ham>spam:
            hb.value=ham
            sb.value=spam
            r.value="Email Is Ham(NOT A SPAM)"
            r.color="#00A80B"
        else:
            hb.value=ham
            sb.value=spam
            r.value="Email Is Ham(NOT A SPAM)"
            r.color="#FF0202"
    ep= ft.Container(
        padding=30,
        bgcolor="#73D4F5",
        border_radius=15,
        content=ft.Column(
            alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
            controls=[
                ft.Row(
                    expand=True,
                    spacing=10,
                    controls=[
                        ft.Text("Spam", size=20, weight=ft.FontWeight.BOLD, color="#FF0202"),
                        sb,
                    ]
                ),
                ft.Row(
                    expand=True,
                    spacing=10,
                    controls=[
                        ft.Text("Ham ", size=20, weight=ft.FontWeight.BOLD, color="#196422"),
                        hb,
                    ]
                ),
            ]
        )
    )
    page.add( ft.Column(
        controls=[ft.Button("Back To Main Page",color="white",bgcolor="blue"),
             email,
             ft.Button("Submit",color="white",bgcolor="blue",on_click=chemail),
             r,
             ep]))
if __name__=="__main__":
    ft.run(main)    